#!/usr/bin/env python3

"""
A professional-grade script for meteor image and video classification, powered by PyTorch.

This script supports two data types:
1. Images (e.g., JPG, PNG) using a 2D CNN.
2. Videos (e.g., WebM, MP4) using a 3D CNN to analyze motion.

It provides parallel modes for each data type (e.g., 'train' vs. 'videotrain').

The 'predict' mode uses an intelligent loading system. By default, it prefers to use
the smaller, clustered models if they are available, falling back to the unclustered
models if not. The --model flag allows the user to override this behavior or select
a single base model from the ensemble for individual prediction.

The 'buildensemble' mode automates the process of training all required models. The
--cluster flag can be used to create additional, highly-compressed versions of the
models using K-Means weight clustering. This mode also includes automatic error
handling for CUDA Out-of-Memory errors, reducing the batch size and retrying.

The --balance flag can be used during training to create synthetic negative samples
(wobbly images or vertically-shifting videos) to address class imbalance.
This balancing operation uses multiprocessing to accelerate file generation on multi-core systems.
"""
import os
import sys
import argparse
import logging
import json
import pathlib
import tempfile
import shutil
import random
import io
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        logging.info("tqdm not found. To see progress bars, please run 'pip install tqdm'")
        return iterable

try:
    from sklearn.cluster import KMeans
except ImportError:
    def KMeans(*args, **kwargs):
        logging.error("scikit-learn not found. To use clustering, please run: pip install scikit-learn")
        sys.exit(1)

try:
    import zstandard as zstd
except ImportError:
    # This placeholder is for script initialization; a real error will be raised if zstd is needed.
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.models as models
import cv2
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

CONFIG = {
    "DEFAULT_IMG_HEIGHT": 96,
    "DEFAULT_IMG_WIDTH": 192,
    "NUM_FRAMES": 16,
    "IMAGE_PARAMS_FILE": 'best_image_params.json',
    "VIDEO_PARAMS_FILE": 'best_video_params.json',
    "STACKER_MODEL_NAME": 'stacker_model.joblib',
    "STACKING_DATA_CSV": 'stacking_train_data.csv',
    # --- Define model configurations for the buildensemble process ---
    "IMAGE_MODEL_CONFIGS": {
        'resnet50': 'meteor_resnet50_model.pth',
        'efficientnet_b0': 'meteor_efficientnet_b0_model.pth',
    },
    "VIDEO_MODEL_CONFIGS": {
        'r2plus1d_18': 'meteor_r2plus1d_18_model.pth',
    },
    # --- Define standalone paths for the custom models ---
    "CUSTOM_IMAGE_MODEL_PATH": 'meteor_custom_image_model.pth',
    "CUSTOM_VIDEO_MODEL_PATH": 'meteor_custom_video_model.pth',
}

def setup_logging(verbose: bool):
    """Configures logging to be quiet by default and verbose when requested."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class ConvNet2D(nn.Module):
    def __init__(self, params: Dict[str, Any], image_size: Tuple[int, int]):
        super(ConvNet2D, self).__init__()
        img_height, img_width = image_size; self.conv_layers = nn.Sequential(); in_channels = 3
        for i in range(params.get('num_conv_layers', 3)):
            out_channels = params.get(f'conv_{i}_filters', 64)
            self.conv_layers.add_module(f'conv2d_{i}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'))
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU()); self.conv_layers.add_module(f'pool2d_{i}', nn.MaxPool2d(2, 2)); in_channels = out_channels
        with torch.no_grad(): flattened_size = self.conv_layers(torch.zeros(1, 3, img_height, img_width)).numel()
        self.fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(flattened_size, params.get('dense_units', 128)), nn.ReLU(),
            nn.Dropout(p=params.get('dropout_rate', 0.5)), nn.Linear(params.get('dense_units', 128), 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.fc_layers(self.conv_layers(x))

class VideoConvNet3D(nn.Module):
    def __init__(self, params: Dict[str, Any], image_size: Tuple[int, int], num_frames: int):
        super(VideoConvNet3D, self).__init__()
        img_height, img_width = image_size; self.conv_layers = nn.Sequential(); in_channels = 3
        for i in range(params.get('num_conv_layers', 3)):
            out_channels = params.get(f'conv_{i}_filters', 32)
            self.conv_layers.add_module(f'conv3d_{i}', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'))
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU()); self.conv_layers.add_module(f'pool3d_{i}', nn.MaxPool3d(kernel_size=(2, 2, 2))); in_channels = out_channels
        with torch.no_grad(): flattened_size = self.conv_layers(torch.zeros(1, 3, num_frames, img_height, img_width)).numel()
        self.fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(flattened_size, params.get('dense_units', 256)), nn.ReLU(),
            nn.Dropout(p=params.get('dropout_rate', 0.5)), nn.Linear(params.get('dense_units', 256), 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.fc_layers(self.conv_layers(x))

def get_model(model_name: str, args: argparse.Namespace, params: Dict[str, Any] = {}) -> nn.Module:
    """
    Factory function to get a model instance by name.
    Supports custom models and pretrained models from torchvision.
    """
    logging.info(f"Initializing model: {model_name}")
    if model_name == 'custom_image':
        return ConvNet2D(params, (args.img_height, args.img_width))
    elif model_name == 'custom_video':
        return VideoConvNet3D(params, (args.img_height, args.img_width), args.num_frames)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        return model
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        return model
    elif model_name == 'r2plus1d_18':
        model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        return model
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def sample_video_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    frames = []; cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
        if total_frames < 1: return frames
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        for i in sorted(set(indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        while len(frames) < num_frames and len(frames) > 0: frames.append(frames[-1])
    finally:
        if cap is not None: cap.release()
    return frames[:num_frames]

class VideoDataset(Dataset):
    def __init__(self, data_dir, num_frames, transform=None):
        self.num_frames, self.transform = num_frames, transform; self.classes, self.class_to_idx = self._find_classes(data_dir); self.samples = self._make_dataset(data_dir, self.class_to_idx)
    def _find_classes(self, dir_path): classes = sorted([d.name for d in os.scandir(dir_path) if d.is_dir()]); return classes, {cls_name: i for i, cls_name in enumerate(classes)}
    def _make_dataset(self, directory, class_to_idx):
        instances, VIDEO_EXTENSIONS = [], ('.webm', '.mp4', '.avi', '.mov', '.mkv')
        for target_class in sorted(class_to_idx.keys()):
            class_index, target_dir = class_to_idx[target_class], os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if fname.lower().endswith(VIDEO_EXTENSIONS): instances.append((os.path.join(root, fname), class_index))
        return instances
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        video_path, target = self.samples[index]; frames = sample_video_frames(video_path, self.num_frames)
        if not frames: return torch.zeros((3, self.num_frames, 1, 1)), target
        if self.transform: frames = [self.transform(frame) for frame in frames]
        return torch.stack(frames, dim=1), target

def _apply_image_wobble(args_tuple):
    """Worker function for multiprocessing to apply wobble effect."""
    input_path, output_path = args_tuple
    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            pixels = np.array(img); height, width, _ = pixels.shape; new_pixels = np.zeros_like(pixels)
            amplitude = random.uniform(8.0, 32.0); frequency = random.uniform(1.0, 4.0)
            for x in range(width):
                shift = int(amplitude * np.sin(2 * np.pi * x * frequency / width))
                rolled_column = np.roll(pixels[:, x, :], shift, axis=0)
                if shift > 0: rolled_column[:shift, :] = 0
                elif shift < 0: rolled_column[shift:, :] = 0
                new_pixels[:, x, :] = rolled_column
            Image.fromarray(new_pixels).save(output_path)
    except Exception as e:
        logging.debug(f"Could not apply image wobble to {input_path}: {e}")

def _apply_video_shift(args_tuple):
    """Worker function for multiprocessing to apply video shift."""
    input_path, output_path, num_frames = args_tuple
    cap, writer = None, None
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return
        fourcc = cv2.VideoWriter_fourcc(*'VP90'); writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        amplitude = random.uniform(8.0, 32.0); frequency = random.uniform(1.0, 4.0)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            shift = int(amplitude * np.sin(2 * np.pi * i * frequency / total_frames)); new_frame = np.zeros_like(frame)
            if shift > 0:
                if shift < height: new_frame[shift:, :, :] = frame[:-shift, :, :]
            elif shift < 0:
                abs_shift = abs(shift)
                if abs_shift < height: new_frame[:-abs_shift, :, :] = frame[abs_shift:, :, :]
            else: new_frame = frame
            writer.write(new_frame)
    except Exception as e:
        logging.debug(f"Could not apply video shift to {input_path}: {e}")
    finally:
        if cap: cap.release()
        if writer: writer.release()

def prepare_data_in_tempdir(temp_dir_path: str, args: argparse.Namespace) -> str:
    """Copies files and performs precise, type-specific balancing using multiprocessing."""
    data_dir = pathlib.Path(temp_dir_path) / 'data'; pos_path = data_dir / 'meteor'; neg_path = data_dir / 'non_meteor'
    pos_path.mkdir(parents=True, exist_ok=True); neg_path.mkdir(parents=True, exist_ok=True)

    logging.info("Organizing original files into temporary directory...")
    shutil.copytree(args.positive_dir, pos_path, dirs_exist_ok=True)
    shutil.copytree(args.negative_dir, neg_path, dirs_exist_ok=True)

    if not getattr(args, 'balance', False):
        logging.info("Skipping dataset balancing."); return str(data_dir)

    def generate_for_type(input_type: str):
        """Counts only relevant file types and generates synthetic data to balance them."""
        is_image = input_type == 'image'
        extensions = ('.jpg', '.jpeg', '.png') if is_image else ('.webm', '.mp4', '.avi', '.mov', '.mkv')
        pos_files_of_type = [f for f in os.listdir(pos_path) if f.lower().endswith(extensions)]
        neg_files_of_type = [f for f in os.listdir(neg_path) if f.lower().endswith(extensions)]
        num_pos_of_type, num_neg_of_type = len(pos_files_of_type), len(neg_files_of_type)
        if num_pos_of_type <= num_neg_of_type:
            logging.info(f"Balancing not needed for '{input_type}': {num_pos_of_type} positives <= {num_neg_of_type} negatives.")
            return

        num_to_generate = num_pos_of_type - num_neg_of_type; max_workers = os.cpu_count() or 1
        logging.info(f"Balancing for '{input_type}': generating {num_to_generate} synthetic negatives using up to {max_workers} processes.")
        source_files = random.choices(pos_files_of_type, k=num_to_generate)
        worker_func = _apply_image_wobble if is_image else _apply_video_shift
        output_prefix = 'synthetic_wobble_' if is_image else 'synthetic_shift_'; output_ext = '.jpg' if is_image else '.webm'
        tasks = []
        for i, source_file_name in enumerate(source_files):
            input_path = os.path.join(pos_path, source_file_name)
            output_path = os.path.join(neg_path, f'{output_prefix}{i:05d}{output_ext}')
            task_args = (input_path, output_path) if is_image else (input_path, output_path, args.num_frames)
            tasks.append(task_args)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(worker_func, tasks), total=len(tasks), desc=f"Balancing ({input_type})"))

    mode_to_check = args.mode
    if mode_to_check in ['buildensemble', 'evaluatestack', 'trainstacker']:
        generate_for_type('image')
        generate_for_type('video')
    elif hasattr(args, 'input_type'):
        generate_for_type(args.input_type)

    return str(data_dir)


def get_dataloaders(args, data_dir, split=True):
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if args.input_type == 'image': full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    else: full_dataset = VideoDataset(data_dir, num_frames=args.num_frames, transform=transform)
    if not hasattr(full_dataset, 'classes') or not full_dataset.classes: raise ValueError(f"No classes found in {data_dir}.")
    num_workers = min(16, os.cpu_count() or 1)
    if not split: return DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True), full_dataset.class_to_idx
    train_size=int(0.8 * len(full_dataset)); val_size = len(full_dataset)-train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    return (DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            full_dataset.class_to_idx)

def _run_training(args, data_dir, model_path, params_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = getattr(args, 'model_name', 'custom_image' if args.input_type == 'image' else 'custom_video')
    logging.info(f"--- Starting Training for '{model_name}' on {device} ---")
    
    params = {};
    if params_file and os.path.exists(params_file):
        with open(params_file, 'r') as f: params = json.load(f)
    if 'custom' in model_name:
        default_params = {'num_conv_layers': 3, 'conv_0_filters': 32, 'conv_1_filters': 64, 'conv_2_filters': 128, 'dense_units': 256, 'dropout_rate': 0.5, 'learning_rate': 1e-4}
    else:
        default_params = {'learning_rate': 1e-4}
    params = {**default_params, **params}
    
    model = get_model(model_name, args, params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate']); criterion = nn.BCELoss()
    
    current_batch_size = args.batch_size
    while current_batch_size >= 1:
        try:
            logging.info(f"Attempting to train with batch size: {current_batch_size}")
            # Create a temporary args namespace to pass the adjusted batch size
            temp_args = argparse.Namespace(**vars(args))
            temp_args.batch_size = current_batch_size
            train_loader, val_loader, _ = get_dataloaders(temp_args, data_dir)
            
            patience, patience_counter, best_val_loss = 8, 0, float('inf')
            for epoch in range(args.epochs):
                model.train()
                for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch"):
                    inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                    optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
                model.eval(); val_loss = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = model(inputs.to(device)); val_loss += criterion(outputs, labels.to(device).float().unsqueeze(1)).item()
                avg_val_loss = val_loss / len(val_loader)
                if avg_val_loss < best_val_loss:
                    best_val_loss, patience_counter = avg_val_loss, 0
                    torch.save(model.state_dict(), model_path)
                    logging.info(f"Epoch {epoch+1}: Val loss for {model_name} improved to {avg_val_loss:.4f}. Model saved.")
                else:
                    patience_counter += 1; logging.info(f"Epoch {epoch+1}: Val loss did not improve. Patience: {patience_counter}/{patience}.")
                if patience_counter >= patience: logging.info("Stopping early."); break
            
            logging.info(f"Training finished successfully for {model_name}. Best model saved to '{model_path}'.")
            return # Exit function on success

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            logging.warning(f"CUDA Out of Memory for '{model_name}' with batch size {current_batch_size}.")
            current_batch_size //= 2
            if current_batch_size < 1:
                logging.error(f"Failed to train '{model_name}' even with batch size 1. Aborting this model's training.")
                break
            logging.info(f"Retrying with a reduced batch size of {current_batch_size}.")

def _run_clustering(model_path: str, args: argparse.Namespace):
    """Loads a trained model, applies K-Means clustering to its weights, and saves the result."""
    logging.info(f"--- Starting K-Means Weight Clustering for {model_path} ---")
    
    model_name = getattr(args, 'model_name', 'unknown')
    
    # 1. Load the dense model
    model = load_model_helper(model_name, model_path.replace('.zst', ''), args)
    
    # 2. Extract weights to cluster
    weights_to_cluster = []
    logging.info("Extracting weights for clustering...")
    for name, param in model.named_parameters():
        if param.dim() > 1:
            weights_to_cluster.append(param.data.flatten())

    if not weights_to_cluster:
        logging.warning("No weights found to cluster. Skipping.")
        return

    all_weights = torch.cat(weights_to_cluster).cpu().numpy().reshape(-1, 1)
    
    # 3. Run K-Means
    logging.info(f"Running K-Means with {args.clusters} clusters on {len(all_weights)} weights...")
    kmeans = KMeans(n_clusters=args.clusters, random_state=0, n_init='auto', max_iter=100).fit(all_weights)
    centroids = kmeans.cluster_centers_.flatten()
    
    # 4. Apply clusters back to a new state_dict
    logging.info("Applying cluster centroids back to the model weights...")
    new_state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            original_weights = param.data.cpu().numpy()
            labels = kmeans.predict(original_weights.reshape(-1, 1))
            clustered_weights = centroids[labels].reshape(original_weights.shape)
            new_state_dict[name] = torch.from_numpy(clustered_weights).to(param.device, dtype=param.dtype)

    # 5. Save and compress the clustered model
    clustered_model_path = model_path.replace('.pth', '_clustered.pth')
    torch.save(new_state_dict, clustered_model_path)
    
    logging.info(f"Compressing clustered model {clustered_model_path} to {clustered_model_path}.zst")
    cctx = zstd.ZstdCompressor(level=19)
    with open(clustered_model_path, 'rb') as f_in, open(clustered_model_path + '.zst', 'wb') as f_out:
        cctx.copy_stream(f_in, f_out)
    
    original_size = os.path.getsize(clustered_model_path) / (1024 * 1024)
    compressed_size = os.path.getsize(clustered_model_path + '.zst') / (1024 * 1024)
    logging.info(f"Clustering and compression complete. Uncompressed: {original_size:.2f} MB, Compressed: {compressed_size:.2f} MB.")
    os.remove(clustered_model_path)


def _run_evaluation(args, data_dir, model_path):
    """Runs evaluation on a single model, calculating and printing key metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = getattr(args, 'model_name', 'unknown')
    logging.info(f"--- Starting Evaluation for '{model_name}' on {device} ---")
    
    model = load_model_helper(model_name, model_path, args).to(device); model.eval()
    loader, class_indices = get_dataloaders(args, data_dir, split=False)
    
    y_true, y_pred_probs = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()
            y_pred_probs.extend(outputs.numpy().flatten()); y_true.extend(labels.numpy().flatten())
    
    y_true, y_pred_probs = np.array(y_true), np.array(y_pred_probs)
    
    y_pred_probs_positive = 1.0 - y_pred_probs if class_indices.get('meteor') == 0 else y_pred_probs
    y_true_positive = (y_true == class_indices.get('meteor')).astype(int)
    
    print("\n" + "="*50 + f"\n--- Evaluation Report for {model_name} ---\n" + "="*50)

    if args.threshold is not None:
        y_pred = (y_pred_probs_positive >= args.threshold).astype(int)
        precision = precision_score(y_true_positive, y_pred, zero_division=0)
        recall = recall_score(y_true_positive, y_pred, zero_division=0)
        f1 = f1_score(y_true_positive, y_pred, zero_division=0)
        
        print(f"✅ Using User-Specified Threshold: {args.threshold:.4f}")
        print(f"\nScores at this threshold:")
        print(f"  - F1-Score:      {f1:.4f}")
        print(f"  - Precision:     {precision:.4f}")
        print(f"  - Recall:        {recall:.4f}\n" + "="*50)
    else:
        precision, recall, thresholds = precision_recall_curve(y_true_positive, y_pred_probs_positive)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]; best_f1 = f1_scores[best_f1_idx]
        best_precision = precision[best_f1_idx]; best_recall = recall[best_f1_idx]
        
        print(f"✅ Optimal Classification Threshold: {best_threshold:.4f}")
        print(f"\nThis threshold provides the best balance (F1-Score) between Precision and Recall:")
        print(f"  - F1-Score:      {best_f1:.4f}")
        print(f"  - Precision:     {best_precision:.4f}")
        print(f"  - Recall:        {best_recall:.4f}\n" + "="*50)

def _run_tuning(args, data_dir, params_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = getattr(args, 'model_name', 'custom_image' if args.input_type == 'image' else 'custom_video')
    logging.info(f"--- Starting Hyperparameter Tuning for {model_name} on {device} ---")
    
    if 'custom' not in model_name:
        logging.warning(f"Tuning is not configured for '{model_name}'; this process is optimized for 'custom' models. Skipping.")
        with open(params_path, 'w') as f: json.dump({'learning_rate': 1e-4}, f, indent=4)
        return

    def objective(trial: optuna.Trial) -> float:
        params = {'num_conv_layers': trial.suggest_int('num_conv_layers', 2, 4), 'dense_units': trial.suggest_categorical('dense_units', [128, 256, 512]),
                  'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5), 'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)}
        for i in range(params['num_conv_layers']): params[f'conv_{i}_filters'] = trial.suggest_categorical(f'conv_{i}_filters', [16, 32, 64, 128])
        try:
            model = get_model(model_name, args, params).to(device)
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate']); criterion = nn.BCELoss()
            train_loader, val_loader, _ = get_dataloaders(args, data_dir); model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad(); loss = criterion(model(inputs), labels); loss.backward(); optimizer.step()
            
            model.eval(); correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs.to(device))
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == labels.to(device).float().unsqueeze(1)).sum().item(); total += labels.size(0)
            return correct / total if total > 0 else 0
        except Exception as e: 
            logging.error(f"Trial failed due to an exception: {e}")
            raise optuna.exceptions.TrialPruned()

    study = optuna.create_study(direction='maximize'); study.optimize(objective, n_trials=args.trials)
    with open(params_path, 'w') as f: json.dump(study.best_params, f, indent=4)
    logging.info(f"Tuning complete. Best validation accuracy: {study.best_value:.4f}. Parameters saved to {params_path}")

def evaluate_stack_mode(args):
    """Evaluates the full stacking ensemble and finds its optimal threshold."""
    logging.info("--- Evaluating Stacking Ensemble ---")
    stacker_model_path = resolve_model_path(args.stacker_model_file, CONFIG["STACKER_MODEL_NAME"])
    if not stacker_model_path:
        logging.error("Stacker model not found. Cannot evaluate."); return
    stacker_model = joblib.load(stacker_model_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        args.balance = False; data_dir = prepare_data_in_tempdir(temp_dir, args)
        
        _, class_to_idx = get_dataloaders(argparse.Namespace(**vars(args), input_type='image'), data_dir, split=False)
        
        data = _get_stacking_data(args, data_dir, class_to_idx)
        
    X, y_true_positive = data.drop('is_meteor', axis=1), data['is_meteor']
    
    X = X[stacker_model.feature_names_in_]
    
    y_pred_probs_positive = stacker_model.predict_proba(X)[:, 1]
    
    print("\n" + "="*50 + "\n--- Evaluation Report for Stacking Ensemble ---\n" + "="*50)

    if args.threshold is not None:
        y_pred = (y_pred_probs_positive >= args.threshold).astype(int)
        precision = precision_score(y_true_positive, y_pred, zero_division=0)
        recall = recall_score(y_true_positive, y_pred, zero_division=0)
        f1 = f1_score(y_true_positive, y_pred, zero_division=0)

        print(f"✅ Using User-Specified Threshold: {args.threshold:.4f}")
        print(f"\nScores at this threshold:")
        print(f"  - F1-Score:      {f1:.4f}")
        print(f"  - Precision:     {precision:.4f}")
        print(f"  - Recall:        {recall:.4f}\n" + "="*50)
    else:
        precision, recall, thresholds = precision_recall_curve(y_true_positive, y_pred_probs_positive)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]; best_f1 = f1_scores[best_f1_idx]
        best_precision = precision[best_f1_idx]; best_recall = recall[best_f1_idx]
        
        print(f"✅ Optimal Classification Threshold: {best_threshold:.4f}")
        print(f"\nThis threshold provides the best balance (F1-Score) between Precision and Recall:")
        print(f"  - F1-Score:      {best_f1:.4f}")
        print(f"  - Precision:     {best_precision:.4f}")
        print(f"  - Recall:        {best_recall:.4f}\n" + "="*50)

    _report_feature_importance(stacker_model)

def run_generic_mode(args):
    """Wrapper for modes that operate on prepared data directories (train, tune, evaluate)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        if 'evaluate' in args.mode: args.balance = False
        data_dir = prepare_data_in_tempdir(temp_dir, args)
        if 'train' in args.mode:
            if args.mode == 'train': args.model_name = 'custom_image'
            if args.mode == 'videotrain': args.model_name = 'custom_video'
            _run_training(args, data_dir, args.output, getattr(args, 'params_file', None))
        elif 'evaluate' in args.mode:
            # Dynamically determine model_name from the provided model file
            filename = os.path.basename(args.model_file)
            # Handles variations like meteor_efficientnet_b0_model_clustered.pth.zst -> efficientnet_b0
            model_name = filename.replace('meteor_', '').replace('_model', '').replace('.pth', '').replace('.zst', '').replace('_clustered', '')
            args.model_name = model_name
            args.input_type = 'video' if model_name in CONFIG["VIDEO_MODEL_CONFIGS"] else 'image'
            logging.info(f"Inferred model name '{model_name}' from file '{filename}'.")
            
            resolved_path = resolve_model_path(None, args.model_file)
            if not resolved_path:
                logging.error(f"Model file not found: {args.model_file}")
                return
            _run_evaluation(args, data_dir, resolved_path.replace('.zst',''))

class PairedDataset(Dataset):
    def __init__(self, file_pairs, args, transform):
        self.file_pairs, self.args, self.transform = file_pairs, args, transform
    def __len__(self): return len(self.file_pairs)
    def __getitem__(self, index):
        img_path, video_path, label = self.file_pairs[index]
        img_tensor = self.transform(Image.open(img_path).convert('RGB'))
        frames = sample_video_frames(video_path, self.args.num_frames)
        video_tensor = torch.zeros((3, self.args.num_frames, self.args.img_height, self.args.img_width))
        if frames: video_tensor = torch.stack([self.transform(f) for f in frames], dim=1)
        return img_tensor, video_tensor, label

def load_model_helper(model_name: str, model_file: str, args: argparse.Namespace) -> nn.Module:
    """Loads a float model's state_dict, handling zstd compression."""
    device = "cpu"
    
    model_path_zst = model_file + '.zst'
    buffer = None
    if os.path.exists(model_path_zst):
        logging.info(f"Decompressing model from {model_path_zst}...")
        with open(model_path_zst, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buffer = io.BytesIO(reader.read())
    elif os.path.exists(model_file):
        buffer = model_file
    else:
        raise FileNotFoundError(f"Could not find model file {model_file} or {model_path_zst}")

    params_path = os.path.splitext(model_file)[0] + '_params.json'
    params = {}
    if os.path.exists(params_path):
        with open(params_path, 'r') as f: params = json.load(f)
    else:
        if 'custom' in model_name:
             logging.warning(f"Parameter file not found for custom model {model_name}.")
    
    model = get_model(model_name, args, params)
    model.to(device)
    model.load_state_dict(torch.load(buffer, map_location=device))
    model.eval()
    return model

def _get_stacking_data(args, data_dir, class_to_idx):
    """
    Generates stacking data by loading FLOAT models.
    """
    device = "cpu"
    logging.info(f"--- Generating stacking data ---")

    image_model_configs = {name: path for name, path in CONFIG["IMAGE_MODEL_CONFIGS"].items()}
    video_model_configs = {name: path for name, path in CONFIG["VIDEO_MODEL_CONFIGS"].items()}
    
    def find_data_pairs(positive_dir, negative_dir):
        pairs = []
        for label_name, directory in [('meteor', positive_dir), ('non_meteor', negative_dir)]:
            label = class_to_idx[label_name]
            for f in os.listdir(directory):
                if label_name == 'non_meteor' and f.startswith('synthetic_'):
                    continue
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    basename = os.path.splitext(f)[0]; img_path = os.path.join(directory, f)
                    vid_path_webm = os.path.join(directory, basename + '.webm')
                    vid_path_mp4 = os.path.join(directory, basename + '.mp4')
                    vid_path = vid_path_webm if os.path.exists(vid_path_webm) else (vid_path_mp4 if os.path.exists(vid_path_mp4) else None)
                    if vid_path: pairs.append((img_path, vid_path, label))
        return pairs

    meteor_dir = os.path.join(data_dir, 'meteor')
    non_meteor_dir = os.path.join(data_dir, 'non_meteor')
    file_pairs = find_data_pairs(meteor_dir, non_meteor_dir)
    
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = PairedDataset(file_pairs, args, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() or 1))
    
    all_predictions = {}
    
    all_model_configs = {**image_model_configs, **video_model_configs}
    for model_name, model_path in all_model_configs.items():
        # Stacking should use the clustered model if it exists, otherwise the fullsize one.
        clustered_path = model_path.replace('.pth', '_clustered.pth')
        resolved_path = resolve_model_path(None, clustered_path)
        if not resolved_path:
            resolved_path = resolve_model_path(None, model_path)
        
        if not resolved_path:
            logging.error(f"Required model not found for stacking: {model_path}. Run buildensemble."); return pd.DataFrame()

        logging.info(f"Loading model for {model_name} from {resolved_path}")
        model = load_model_helper(model_name, resolved_path.replace('.zst',''), args)
        
        preds_list = []
        with torch.no_grad():
            data_iterator = ( (img_batch, vid_batch) for img_batch, vid_batch, _ in loader )
            for img_batch, vid_batch in tqdm(data_iterator, desc=f"Predicting with {model_name}", total=len(loader)):
                inputs = vid_batch if model_name in video_model_configs else img_batch
                preds = model(inputs.to(device)).cpu().numpy().flatten()
                prob_positive = 1.0 - preds if class_to_idx.get('meteor') == 0 else preds
                preds_list.extend(prob_positive)
        all_predictions[f"pred_{model_name}"] = preds_list

    true_labels = [label for _, _, label in loader]
    true_labels = torch.cat(true_labels).numpy().flatten()

    df_data = pd.DataFrame(all_predictions)
    df_data['is_meteor_label'] = true_labels
    df_data['is_meteor'] = (df_data['is_meteor_label'] == class_to_idx['meteor']).astype(int)
    df_data = df_data.drop('is_meteor_label', axis=1)

    logging.info(f"Saving stacking data to '{CONFIG['STACKING_DATA_CSV']}'")
    df_data.to_csv(CONFIG['STACKING_DATA_CSV'], index=False)
    
    return df_data


def _report_feature_importance(stacker_model: LogisticRegression):
    """Analyzes and prints the importance of each model in the stacker."""
    if not hasattr(stacker_model, 'coef_'):
        logging.warning("Stacker model does not have coefficients to report (not a linear model).")
        return
    
    feature_names = stacker_model.feature_names_in_
    coefficients = stacker_model.coef_[0]
    
    importance_df = pd.DataFrame({'Model': feature_names, 'Coefficient': coefficients})
    importance_df['Absolute Coefficient'] = np.abs(importance_df['Coefficient'])
    importance_df = importance_df.sort_values(by='Absolute Coefficient', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*60)
    print("--- Stacking Ensemble: Model Importance Report ---")
    print("="*60)
    print("Models are ranked by the absolute weight the final classifier")
    print("assigns to their predictions. A low absolute coefficient")
    print("suggests a model may be redundant or less impactful.")
    print("-" * 60)
    print(importance_df[['Model', 'Coefficient']].to_string())
    print("-" * 60)

    low_importance_models = importance_df[importance_df['Absolute Coefficient'] < 0.1]
    if not low_importance_models.empty:
        print("\n💡 Recommendation:")
        print("The following models have very low importance and could potentially be removed")
        print("to simplify the ensemble without a significant loss in accuracy:")
        for model_name in low_importance_models['Model']:
            print(f"  - {model_name.replace('pred_', '')}")
    else:
        print("\n✅ All models appear to contribute meaningfully to the ensemble.")
    print("="*60)

def resolve_model_path(arg_path: str, default_name: str) -> Optional[str]:
    """Finds a model file, respecting user overrides and searching for .pth and .pth.zst."""
    search_names = []
    if arg_path:
        search_names.extend([arg_path, arg_path + '.zst'])
    
    search_names.extend([default_name, default_name + '.zst'])

    search_names = list(OrderedDict.fromkeys(search_names))
    
    for name in search_names:
        if os.path.exists(os.path.abspath(name)):
            logging.info(f"Found model file in current directory: ./{name}")
            return name
        user_model_dir = pathlib.Path.home() / 'nmn' / 'model'
        user_model_path = user_model_dir / name
        if os.path.exists(user_model_path):
            logging.info(f"Found model file in user directory: {user_model_path}")
            return str(user_model_path)
    return None

def find_and_load_model(model_name: str, args: argparse.Namespace) -> Optional[nn.Module]:
    """Intelligently finds and loads the best available model based on the --model flag."""
    base_path = f"meteor_{model_name}_model"
    clustered_path = base_path + "_clustered.pth"
    unclustered_path = base_path + ".pth"

    path_to_load = None
    
    if args.model == 'fullsize':
        logging.info(f"User override: searching for unclustered model for {model_name}.")
        resolved_unclustered = resolve_model_path(None, unclustered_path)
        if resolved_unclustered:
            path_to_load = resolved_unclustered.replace('.zst', '')
    else: # Default behavior
        resolved_clustered = resolve_model_path(None, clustered_path)
        if resolved_clustered:
            logging.info(f"Clustered model found for {model_name}, using it.")
            path_to_load = resolved_clustered.replace('.zst', '')
        else:
            logging.info(f"Clustered model not found for {model_name}, falling back to unclustered version.")
            resolved_unclustered = resolve_model_path(None, unclustered_path)
            if resolved_unclustered:
                path_to_load = resolved_unclustered.replace('.zst', '')

    if not path_to_load:
        logging.warning(f"Could not find any suitable model file for {model_name}.")
        return None
        
    return load_model_helper(model_name, path_to_load, args)


def run_ensemble_prediction(args: argparse.Namespace, device: str):
    """Loads the stacking ensemble and predicts using a fallback model strategy."""
    logging.info("--- Using Stacking Ensemble for Prediction ---")
    stacker_model_path = resolve_model_path(None, CONFIG["STACKER_MODEL_NAME"])
    if not stacker_model_path:
        logging.error(f"Stacker model '{CONFIG['STACKER_MODEL_NAME']}' not found. Cannot use ensemble mode."); return
    
    stacker_model = joblib.load(stacker_model_path)
    
    image_models = {
        'resnet50': find_and_load_model('resnet50', args),
        'efficientnet_b0': find_and_load_model('efficientnet_b0', args),
    }
    video_models = {
        'r2plus1d_18': find_and_load_model('r2plus1d_18', args),
    }
    
    all_models = {**image_models, **video_models}
    if any(model is None for model in all_models.values()):
        logging.error("One or more base models for the ensemble could not be found. Cannot make predictions."); return

    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for file_path in args.files_to_predict:
        basename, _ = os.path.splitext(file_path)
        img_path = next((f for f in [f"{basename}.jpg", f"{basename}.jpeg", f"{basename}.png"] if os.path.exists(f)), None)
        vid_path = next((f for f in [f"{basename}.webm", f"{basename}.mp4", f"{basename}.avi", f"{basename}.mov", f"{basename}.mkv"] if os.path.exists(f)), None)

        if not (img_path and vid_path):
            logging.warning(f"Could not find a matching image/video pair for '{basename}'. Skipping."); continue

        try:
            img_tensor = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            frames = sample_video_frames(vid_path, args.num_frames)
            if not frames:
                logging.warning(f"Could not extract frames from video: {vid_path}. Skipping."); continue
            vid_tensor = torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)

            with torch.no_grad():
                base_predictions = {}
                for name, model in image_models.items():
                    pred = model(img_tensor).item(); base_predictions[f"pred_{name}"] = 1.0 - pred
                for name, model in video_models.items():
                    pred = model(vid_tensor).item(); base_predictions[f"pred_{name}"] = 1.0 - pred
            
            prediction_df = pd.DataFrame([base_predictions])[stacker_model.feature_names_in_]
            final_prob = stacker_model.predict_proba(prediction_df)[0][1]
            
            print(f"{os.path.basename(basename)}: {final_prob:.6f}")

        except Exception as e:
            logging.error(f"Failed to process '{file_path}': {e}")


def run_custom_prediction(args: argparse.Namespace, device: str):
    """Loads only the custom models and predicts by averaging their probabilities."""
    logging.info("--- Using Custom Models for Prediction ---")
    
    img_model_path = resolve_model_path(None, CONFIG["CUSTOM_IMAGE_MODEL_PATH"])
    vid_model_path = resolve_model_path(None, CONFIG["CUSTOM_VIDEO_MODEL_PATH"])
    
    if not (img_model_path and vid_model_path):
        logging.error("One or both custom models not found. Cannot make prediction in 'custom' mode."); return

    image_model = load_model_helper('custom_image', img_model_path.replace('.zst',''), args)
    video_model = load_model_helper('custom_video', vid_model_path.replace('.zst',''), args)
    
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for file_path in args.files_to_predict:
        basename, _ = os.path.splitext(file_path)
        img_path = next((f for f in [f"{basename}.jpg", f"{basename}.jpeg", f"{basename}.png"] if os.path.exists(f)), None)
        vid_path = next((f for f in [f"{basename}.webm", f"{basename}.mp4", f"{basename}.avi", f"{basename}.mov", f"{basename}.mkv"] if os.path.exists(f)), None)

        if not (img_path and vid_path):
            logging.warning(f"Could not find a matching image/video pair for '{basename}'. Skipping."); continue
        
        try:
            img_tensor = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            frames = sample_video_frames(vid_path, args.num_frames)
            if not frames:
                logging.warning(f"Could not extract frames from video: {vid_path}. Skipping."); continue
            vid_tensor = torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                img_pred = 1.0 - image_model(img_tensor).item()
                vid_pred = 1.0 - video_model(vid_tensor).item()
            
            final_prob = (img_pred + vid_pred) / 2.0
            print(f"{os.path.basename(basename)}: {final_prob:.6f}")

        except Exception as e:
            logging.error(f"Failed to process '{file_path}': {e}")

def run_single_model_prediction(args: argparse.Namespace, device: str):
    """Loads and predicts using a single specified ensemble model."""
    model_name = args.model
    logging.info(f"--- Using SINGLE MODEL ({model_name}) for Prediction ---")

    # The 'find_and_load_model' function handles the logic of finding clustered vs. unclustered
    model = find_and_load_model(model_name, args)
    if model is None:
        logging.error(f"Could not load the {model_name} model. Aborting."); return

    is_video_model = model_name in CONFIG["VIDEO_MODEL_CONFIGS"]

    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for file_path in args.files_to_predict:
        basename, _ = os.path.splitext(file_path)
        
        try:
            if is_video_model:
                vid_path = next((f for f in [f"{basename}.webm", f"{basename}.mp4", f"{basename}.avi", f"{basename}.mov", f"{basename}.mkv"] if os.path.exists(f)), None)
                if not vid_path:
                    logging.warning(f"Could not find a video file for '{basename}'. Skipping."); continue
                frames = sample_video_frames(vid_path, args.num_frames)
                if not frames:
                    logging.warning(f"Could not extract frames from video: {vid_path}. Skipping."); continue
                tensor = torch.stack([transform(f) for f in frames], dim=1)
                if tensor.dim() == 4: # Add batch dimension
                    tensor = tensor.unsqueeze(0)
                # Permute from (B, F, C, H, W) to (B, C, F, H, W) for the model
                tensor = tensor.permute(0, 2, 1, 3, 4).to(device)
            else: # Image model
                img_path = next((f for f in [f"{basename}.jpg", f"{basename}.jpeg", f"{basename}.png"] if os.path.exists(f)), None)
                if not img_path:
                    logging.warning(f"Could not find an image file for '{basename}'. Skipping."); continue
                tensor = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_prob = model(tensor).item()
            
            # Models are trained to output P(non_meteor), so P(meteor) is 1.0 - P(non_meteor)
            final_prob = 1.0 - pred_prob
            print(f"{os.path.basename(basename)}: {final_prob:.6f}")

        except Exception as e:
            logging.error(f"Failed to process '{file_path}': {e}")


def predict_mode(args: argparse.Namespace):
    """Routes prediction to the appropriate handler based on the --model flag."""
    device = "cpu"
    logging.info(f"--- Starting Prediction Mode on {device} (CPU is used for compatibility) ---")
    logging.info(f"--- Model preference set to: {args.model} ---")

    if args.model == 'custom':
        run_custom_prediction(args, device)
    elif args.model in CONFIG['IMAGE_MODEL_CONFIGS'] or args.model in CONFIG['VIDEO_MODEL_CONFIGS']:
        run_single_model_prediction(args, device)
    else: # 'default' or 'fullsize'
        run_ensemble_prediction(args, device)


def _process_single_model(task_args):
    """Worker function to train and cluster a single model. OOM errors are handled by _run_training."""
    model_name, model_path, args, data_dir = task_args
    
    final_artifact_path = model_path
    if getattr(args, 'cluster', False):
        final_artifact_path = model_path.replace('.pth', '_clustered.pth')

    if args.resume and resolve_model_path(None, final_artifact_path) is not None:
        logging.info(f"Resuming: Found existing final model for '{model_name}'. Skipping.")
        return f"{model_name}: Skipped (resume)"

    train_args_dict = vars(args).copy()
    train_args_dict['model_name'] = model_name
    train_args_dict['input_type'] = 'video' if model_name in CONFIG["VIDEO_MODEL_CONFIGS"] else 'image'
    train_args = argparse.Namespace(**train_args_dict)

    try:
        # _run_training now handles its own OOM errors and will not raise them.
        # It will either succeed and save a model, or fail and log an error.
        if not resolve_model_path(None, model_path):
             _run_training(train_args, data_dir, model_path, params_file=None)

        # Proceed to clustering only if the base model was created successfully.
        if getattr(args, 'cluster', False) and resolve_model_path(None, model_path):
            clustered_path = model_path.replace('.pth', '_clustered.pth')
            if not resolve_model_path(None, clustered_path):
                _run_clustering(model_path, train_args)
        
        # Verify that the final expected file exists to confirm success.
        if resolve_model_path(None, final_artifact_path):
            return f"{model_name}: Success"
        else:
            return f"{model_name}: Failed (likely OOM)"

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {model_name}: {e}", exc_info=True)
        return f"{model_name}: Failed (Exception)"

def build_ensemble_mode(args):
    """Automates the entire process of tuning and training all models in parallel."""
    logging.info("--- 🚀 Starting Full Ensemble Build Process 🚀 ---")
    if args.resume:
        logging.info("--- Resuming previous build ---")
    logging.warning("This is a long-running process that will train multiple advanced models.")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_in_tempdir(temp_dir, args)
        
        tasks = []
        all_models = {**CONFIG["IMAGE_MODEL_CONFIGS"], **CONFIG["VIDEO_MODEL_CONFIGS"]}
        for model_name, model_path in all_models.items():
            tasks.append((model_name, model_path, args, data_dir))
            
        # Limit to 3 workers since that's the number of models
        max_workers = min(3, os.cpu_count() or 1)
        logging.info(f"Processing {len(tasks)} models in parallel with up to {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(_process_single_model, tasks), total=len(tasks), desc="Building models"))
        
        logging.info("Model processing summary:")
        for res in results:
            logging.info(f"  - {res}")
            
    logging.info("\n--- Building Stacking Meta-Learner ---")
    train_stacker_args = argparse.Namespace(**vars(args), output=args.stacker_model_file)
    train_stacker_mode(train_stacker_args)
    
    tune_stacker_args = argparse.Namespace(input_csv=CONFIG['STACKING_DATA_CSV'], output=args.stacker_model_file)
    tune_stacker_mode(tune_stacker_args)

    logging.info("--- ✅ Full Ensemble Build Complete! ✅ ---")
    logging.info("Loading final tuned stacker model to show importance report...")
    final_stacker_model = joblib.load(args.stacker_model_file)
    _report_feature_importance(final_stacker_model)

def train_stacker_mode(args):
    """Trains the stacking meta-learner, using a cached data file if available."""
    logging.info("--- Training Stacking Meta-Learner ---")

    data = None
    if getattr(args, 'resume', False) and os.path.exists(CONFIG['STACKING_DATA_CSV']):
        logging.info(f"Found existing stacking data at '{CONFIG['STACKING_DATA_CSV']}' and --resume is active. Reusing it.")
        data = pd.read_csv(CONFIG['STACKING_DATA_CSV'])
    else:
        if getattr(args, 'resume', False):
            logging.info(f"Resume is active, but no cached stacking data found. Generating from scratch...")
        else:
            logging.info(f"--resume not specified. Regenerating stacking data from scratch...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = prepare_data_in_tempdir(temp_dir, args)
            _, class_to_idx = get_dataloaders(argparse.Namespace(**vars(args), input_type='image'), data_dir, split=False)
            
            data = _get_stacking_data(args, data_dir, class_to_idx)
            if data.empty:
                logging.error("Stacking data could not be generated. Aborting stacker training.")
                return

    logging.info(f"Using {len(data)} samples to train meta-learner...")

    X = data.drop('is_meteor', axis=1)
    y = data['is_meteor']
    
    meta_learner = LogisticRegression(solver='liblinear')
    meta_learner.fit(X, y)
    joblib.dump(meta_learner, args.output)
    logging.info(f"Stacking meta-learner trained and saved to {args.output}")

def tune_stacker_mode(args):
    """Tunes the stacking meta-learner on existing stacking data."""
    logging.info("--- Tuning Stacking Meta-Learner ---")
    if not os.path.exists(args.input_csv):
        logging.error(f"Stacking data file not found: {args.input_csv}. Run 'trainstacker' or 'buildensemble' first.")
        return

    data = pd.read_csv(args.input_csv)
    X = data.drop('is_meteor', axis=1)
    y = data['is_meteor']
    
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X, y)
    
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    joblib.dump(grid_search.best_estimator_, args.output)
    logging.info(f"Tuned stacking meta-learner saved to {args.output}")


def main():
    """Parses arguments and dispatches to the correct mode function."""
    parser = argparse.ArgumentParser(description="A tool for meteor image and video classification.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose logging.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Operating mode.")
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--img-width', type=int, default=CONFIG["DEFAULT_IMG_WIDTH"])
    parent_parser.add_argument('--img-height', type=int, default=CONFIG["DEFAULT_IMG_HEIGHT"])
    parent_parser.add_argument('--batch-size', type=int, default=16)
    
    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('positive_dir', help="Directory of positive samples ('meteor').")
    dir_parent_parser.add_argument('negative_dir', help="Directory of negative samples ('non_meteor').")
    
    video_parent_parser = argparse.ArgumentParser(add_help=False)
    video_parent_parser.add_argument('--num-frames', type=int, default=CONFIG["NUM_FRAMES"])
    
    balance_parent_parser = argparse.ArgumentParser(add_help=False)
    balance_parent_parser.add_argument('--balance', action='store_true', help="Enable dataset balancing by creating synthetic negatives.")
    balance_parent_parser.add_argument('--cluster', action='store_true', help="Apply K-Means weight clustering after training.")
    balance_parent_parser.add_argument('--clusters', type=int, default=256, help="Number of clusters (bins) for weight clustering.")

    p_train = subparsers.add_parser('train', help="Train the custom image model.", parents=[parent_parser, dir_parent_parser, balance_parent_parser])
    p_train.add_argument('--params-file', help="Path to a .json file with hyperparameters."); p_train.add_argument('--epochs', type=int, default=50); p_train.add_argument('-o', '--output', default=CONFIG["CUSTOM_IMAGE_MODEL_PATH"])
    
    p_vtrain = subparsers.add_parser('videotrain', help="Train the custom video model.", parents=[parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    p_vtrain.add_argument('--params-file', help="Path to a .json file with hyperparameters."); p_vtrain.add_argument('--epochs', type=int, default=50); p_vtrain.add_argument('-o', '--output', default=CONFIG["CUSTOM_VIDEO_MODEL_PATH"])
    
    p_eval = subparsers.add_parser('evaluate', help="Evaluate an image or video model.", parents=[parent_parser, dir_parent_parser, video_parent_parser])
    p_eval.add_argument('-m', '--model-file', default=CONFIG["CUSTOM_IMAGE_MODEL_PATH"])
    p_eval.add_argument('--threshold', type=float, help="Use a specific threshold for evaluation instead of finding the optimal one.")
    
    p_veval = subparsers.add_parser('videoevaluate', help="Evaluate the custom video model.", parents=[parent_parser, dir_parent_parser, video_parent_parser])
    p_veval.add_argument('-m', '--model-file', default=CONFIG["CUSTOM_VIDEO_MODEL_PATH"])
    
    p_tune = subparsers.add_parser('tune', help="Tune the custom image model.", parents=[parent_parser, dir_parent_parser, balance_parent_parser])
    p_tune.add_argument('--trials', type=int, default=20); p_tune.add_argument('-o', '--output', default=CONFIG["IMAGE_PARAMS_FILE"])

    p_vtune = subparsers.add_parser('videotune', help="Tune the custom video model.", parents=[parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    p_vtune.add_argument('--trials', type=int, default=20); p_vtune.add_argument('-o', '--output', default=CONFIG["VIDEO_PARAMS_FILE"])

    p_build = subparsers.add_parser('buildensemble', help="Run the full end-to-end training for all models and the ensemble.", parents=[parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    p_build.add_argument('--stacker-model-file', default=CONFIG["STACKER_MODEL_NAME"])
    p_build.add_argument('--epochs', type=int, default=50)
    p_build.add_argument('--resume', action='store_true', help="Skip training for models with existing output files.")

    p_stack_train = subparsers.add_parser('trainstacker', help="Create data & train the meta-learner (assumes base models exist).", parents=[parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    p_stack_train.add_argument('-o', '--output', default=CONFIG["STACKER_MODEL_NAME"])

    p_stack_tune = subparsers.add_parser('tunestack', help="Tune the meta-learner (assumes stacking data CSV exists).", parents=[])
    p_stack_tune.add_argument('--input-csv', default=CONFIG["STACKING_DATA_CSV"], help="Path to the CSV file with stacking data."); p_stack_tune.add_argument('-o', '--output', default=CONFIG["STACKER_MODEL_NAME"])
    
    p_stack_eval = subparsers.add_parser('evaluatestack', help="Evaluate the full stacking ensemble.", parents=[parent_parser, dir_parent_parser, video_parent_parser])
    p_stack_eval.add_argument('--stacker-model-file', default=CONFIG["STACKER_MODEL_NAME"])
    p_stack_eval.add_argument('--threshold', type=float, help="Use a specific threshold for evaluation instead of finding the optimal one.")

    predict_model_choices = ['default', 'fullsize', 'custom'] + list(CONFIG['IMAGE_MODEL_CONFIGS'].keys()) + list(CONFIG['VIDEO_MODEL_CONFIGS'].keys())
    p_predict = subparsers.add_parser('predict', help="Classify files using an intelligent, fallback-enabled model system.", parents=[parent_parser, video_parent_parser])
    p_predict.add_argument('files_to_predict', nargs='+', help="Path to image or video file(s) to classify. The script will find matching pairs.")
    p_predict.add_argument('--model', choices=predict_model_choices, default='default', 
                           help="Model preference: 'default' prefers clustered models, 'fullsize' forces unclustered models, 'custom' uses basic standalone models, or specify a single base model name (e.g., 'resnet50').")

    args = parser.parse_args()
    setup_logging(args.verbose)
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, running on CPU. This will be very slow for training.")
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        logging.warning("MPS is not available for accelerated training on Apple Silicon.")
    
    # --- Mode Dispatcher ---
    if args.mode == 'predict':
        predict_mode(args)
    elif args.mode == 'buildensemble':
        build_ensemble_mode(args)
    elif args.mode == 'trainstacker':
        train_stacker_mode(args)
    elif args.mode == 'tunestack':
        tune_stacker_mode(args)
    elif args.mode == 'evaluatestack':
        evaluate_stack_mode(args)
    elif 'tune' in args.mode:
        args.input_type = 'video' if 'video' in args.mode else 'image'
        if args.mode == 'tune': args.model_name = 'custom_image'
        if args.mode == 'videotune': args.model_name = 'custom_video'
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = prepare_data_in_tempdir(temp_dir, args)
            _run_tuning(args, data_dir, args.output)
    else:
        run_generic_mode(args)

if __name__ == '__main__':
    # Set start method for multiprocessing to be compatible with CUDA and some OSes
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    main()
