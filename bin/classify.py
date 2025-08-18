#!/usr/bin/env python3

"""
A professional-grade script for meteor image and video classification, powered by PyTorch.

This script supports two data types:
1. Images (e.g., JPG, PNG) using a 2D CNN.
2. Videos (e.g., WebM, MP4) using a 3D CNN to analyze motion.

It provides parallel modes for each data type (e.g., 'train' vs. 'videotrain').
The 'predict' mode is unified and intelligently uses the best available model.

The 'buildensemble' mode automates the entire process of training multiple advanced models
(including ResNet, EfficientNet, and R(2+1)D) and combining them into a final,
high-accuracy stacking ensemble. It concludes by reporting on the importance of each
model, allowing for future optimization. A --resume flag allows this long-running
process to be stopped and restarted without redoing completed work.

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
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        logging.info("tqdm not found. To see progress bars, please run 'pip install tqdm'")
        return iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.models as models
import cv2
from sklearn.metrics import precision_recall_curve
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

    # Determine which types to generate based on the execution mode
    mode_to_check = args.mode
    if mode_to_check in ['buildensemble', 'evaluatestack', 'trainstacker']:
        generate_for_type('image')
        generate_for_type('video')
    elif hasattr(args, 'input_type'):
        generate_for_type(args.input_type)
        
    return str(data_dir)


def get_dataloaders(args, data_dir, split=True):
    # Pre-trained models require normalization as per torchvision standards
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
    device = "cuda" if torch.cuda.is_available() else "cpu"; model_name = getattr(args, 'model_name', 'custom_image' if args.input_type == 'image' else 'custom_video')
    logging.info(f"--- Starting Training for '{model_name}' on {device} ---")
    params = {};
    if params_file and os.path.exists(params_file):
        with open(params_file, 'r') as f: params = json.load(f)
    # Use good defaults for fine-tuning pre-trained models, otherwise use custom model defaults
    if 'custom' in model_name:
        default_params = {'num_conv_layers': 3, 'conv_0_filters': 32, 'conv_1_filters': 64, 'conv_2_filters': 128, 'dense_units': 256, 'dropout_rate': 0.5, 'learning_rate': 1e-4}
    else: # Fine-tuning defaults
        default_params = {'learning_rate': 1e-4}
    params = {**default_params, **params} # User params override defaults
    
    train_loader, val_loader, _ = get_dataloaders(args, data_dir)
    model = get_model(model_name, args, params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate']); criterion = nn.BCELoss()
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
            best_val_loss, patience_counter = avg_val_loss, 0; torch.save(model.state_dict(), model_path)
            logging.info(f"Epoch {epoch+1}: Val loss for {model_name} improved to {avg_val_loss:.4f}. Model saved.")
        else:
            patience_counter += 1; logging.info(f"Epoch {epoch+1}: Val loss did not improve. Patience: {patience_counter}/{patience}.")
        if patience_counter >= patience: logging.info("Stopping early."); break
    logging.info(f"Training finished for {model_name}. Best model saved to '{model_path}'.")

def _run_evaluation(args, data_dir, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = getattr(args, 'model_name', 'custom_image' if args.input_type == 'image' else 'custom_video')
    logging.info(f"--- Starting Evaluation for '{model_name}' on {device} ---")
    if not os.path.isfile(model_path): logging.error(f"Model file not found: '{model_path}'"); return
    
    model = load_model_helper(model_name, model_path, args).to(device); model.eval()
    loader, class_indices = get_dataloaders(args, data_dir, split=False)
    
    y_true, y_pred_probs = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            outputs = model(inputs.to(device)).cpu()
            y_pred_probs.extend(outputs.numpy().flatten()); y_true.extend(labels.numpy().flatten())
    y_true, y_pred_probs = np.array(y_true), np.array(y_pred_probs)
    
    # Class index 0 is 'meteor', so a low output score from sigmoid means high confidence of meteor.
    # We invert the probability to represent P(meteor).
    y_pred_probs_positive = 1.0 - y_pred_probs if class_indices.get('meteor') == 0 else y_pred_probs
    y_true_positive = (y_true == class_indices.get('meteor')).astype(int)
    
    precision, recall, thresholds = precision_recall_curve(y_true_positive, y_pred_probs_positive)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]; best_f1 = f1_scores[best_f1_idx]
    best_precision = precision[best_f1_idx]; best_recall = recall[best_f1_idx]
    
    print("\n" + "="*50 + f"\n--- Evaluation Report for {model_name} ---\n" + "="*50)
    print(f"✅ Optimal Classification Threshold: {best_threshold:.4f}")
    print(f"\nThis threshold provides the best balance (F1-Score) between Precision and Recall:")
    print(f"  - F1-Score:      {best_f1:.4f}")
    print(f"  - Precision:     {best_precision:.4f}")
    print(f"  - Recall:        {best_recall:.4f}\n" + "="*50)

def _run_tuning(args, data_dir, params_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = getattr(args, 'model_name', 'custom_image' if args.input_type == 'image' else 'custom_video')
    logging.info(f"--- Starting Hyperparameter Tuning for {model_name} on {device} ---")
    
    # This tuning objective is primarily for the custom models.
    if 'custom' not in model_name:
        logging.warning(f"Tuning is not configured for '{model_name}'; this process is optimized for 'custom' models. Skipping.")
        # Create a dummy params file with good defaults
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
            # Run a single epoch for a quick evaluation
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
        
        # Determine the class mapping from the data directory
        _, class_to_idx = get_dataloaders(argparse.Namespace(**vars(args), input_type='image'), data_dir, split=False)
        
        # To evaluate, we need predictions from the base models that the stacker was trained on.
        # We find the paths to these models.
        image_model_configs = {name: resolve_model_path(None, path) for name, path in CONFIG["IMAGE_MODEL_CONFIGS"].items()}
        video_model_configs = {name: resolve_model_path(None, path) for name, path in CONFIG["VIDEO_MODEL_CONFIGS"].items()}

        if any(v is None for v in image_model_configs.values()) or any(v is None for v in video_model_configs.values()):
            logging.error("One or more base models for the ensemble were not found. Cannot generate data for evaluation."); return

        data = _get_stacking_data(args, image_model_configs, video_model_configs, data_dir, class_to_idx)
        
    X, y_true_positive = data.drop('is_meteor', axis=1), data['is_meteor']
    
    # Ensure columns are in the same order as when the model was trained
    X = X[stacker_model.feature_names_in_]
    
    y_pred_probs_positive = stacker_model.predict_proba(X)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_true_positive, y_pred_probs_positive)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]; best_f1 = f1_scores[best_f1_idx]
    best_precision = precision[best_f1_idx]; best_recall = recall[best_f1_idx]
    
    print("\n" + "="*50 + "\n--- Evaluation Report for Stacking Ensemble ---\n" + "="*50)
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
            # Set a default model name for generic train modes
            if args.mode == 'train': args.model_name = 'custom_image'
            if args.mode == 'videotrain': args.model_name = 'custom_video'
            _run_training(args, data_dir, args.output, getattr(args, 'params_file', None))
        elif 'evaluate' in args.mode:
            if args.mode == 'evaluate': args.model_name = 'custom_image'
            if args.mode == 'videoevaluate': args.model_name = 'custom_video'
            _run_evaluation(args, data_dir, args.model_file)

class PairedDataset(Dataset):
    def __init__(self, file_pairs, args, transform):
        self.file_pairs, self.args, self.transform = file_pairs, args, transform
    def __len__(self): return len(self.file_pairs)
    def __getitem__(self, index):
        img_path, video_path, label = self.file_pairs[index]
        img_tensor = self.transform(Image.open(img_path).convert('RGB'))
        frames = sample_video_frames(video_path, self.args.num_frames)
        # Use a normalized tensor for empty videos to match transform
        video_tensor = torch.zeros((3, self.args.num_frames, self.args.img_height, self.args.img_width))
        if frames: video_tensor = torch.stack([self.transform(f) for f in frames], dim=1)
        return img_tensor, video_tensor, label

def load_model_helper(model_name: str, model_file: str, args: argparse.Namespace) -> nn.Module:
    """Loads a model's state dict, supporting various architectures."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- FIX: Use consistent parameter file naming ---
    params_path = os.path.splitext(model_file)[0] + '_params.json'
    params = {}

    if os.path.exists(params_path):
        logging.info(f"Found parameter file for {model_name}: {params_path}")
        with open(params_path, 'r') as f: params = json.load(f)
    else:
        # This is expected for pre-trained models, so a warning is appropriate.
        if 'custom' in model_name:
             logging.warning(f"Parameter file not found for custom model {model_name} at {params_path}.")
        else:
             logging.info(f"No parameter file needed for pre-trained model {model_name}. Using default architecture.")

    model = get_model(model_name, args, params).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device)); model.eval()
    return model

def _get_stacking_data(args, image_model_configs, video_model_configs, data_dir, class_to_idx):
    """Helper to generate predictions from all base models for stacking."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # Get Image Model Predictions
    for model_name, model_path in image_model_configs.items():
        logging.info(f"Generating predictions for image model: {model_name}")
        model = load_model_helper(model_name, model_path, args)
        preds_list = []
        with torch.no_grad():
            for img_batch, _, _ in tqdm(loader, desc=f"Predicting with {model_name}"):
                preds = model(img_batch.to(device)).cpu().numpy().flatten()
                prob_positive = 1.0 - preds if class_to_idx.get('meteor') == 0 else preds
                preds_list.extend(prob_positive)
        all_predictions[f"pred_{model_name}"] = preds_list
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Get Video Model Predictions
    for model_name, model_path in video_model_configs.items():
        logging.info(f"Generating predictions for video model: {model_name}")
        model = load_model_helper(model_name, model_path, args)
        preds_list = []
        with torch.no_grad():
            for _, vid_batch, _ in tqdm(loader, desc=f"Predicting with {model_name}"):
                preds = model(vid_batch.to(device)).cpu().numpy().flatten()
                prob_positive = 1.0 - preds if class_to_idx.get('meteor') == 0 else preds
                preds_list.extend(prob_positive)
        all_predictions[f"pred_{model_name}"] = preds_list
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Get true labels
    true_labels = []
    for _, _, label_batch in loader:
        true_labels.extend(label_batch.numpy().flatten())

    df_data = pd.DataFrame(all_predictions)
    df_data['is_meteor_label'] = true_labels
    df_data['is_meteor'] = (df_data['is_meteor_label'] == class_to_idx['meteor']).astype(int)
    return df_data.drop('is_meteor_label', axis=1)


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
    """Finds a model file, respecting user overrides and searching default locations."""
    if arg_path and arg_path != default_name:
        if os.path.exists(arg_path):
            logging.info(f"Using user-specified model file: {arg_path}")
            return arg_path
        logging.error(f"User-specified model file not found: '{arg_path}'. Cannot proceed.")
        return None
    
    # Search in current directory then in user model directory
    if os.path.exists(default_name):
        logging.info(f"Found model file in current directory: ./{default_name}")
        return default_name
    user_model_dir = pathlib.Path.home() / 'nmn' / 'model'
    user_model_path = user_model_dir / default_name
    if os.path.exists(user_model_path):
        logging.info(f"Found model file in user directory: {user_model_path}")
        return str(user_model_path)
    return None

def predict_mode(args):
    """Classifies files using the full, trained stacking ensemble."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"--- Starting Prediction Mode on {device} ---")

    # 1. Find and load all necessary models
    stacker_model_path = resolve_model_path(None, CONFIG["STACKER_MODEL_NAME"])
    if not stacker_model_path:
        logging.error(f"Stacker model '{CONFIG['STACKER_MODEL_NAME']}' not found. Please run 'buildensemble' first."); return
    
    image_model_configs = {name: resolve_model_path(None, path) for name, path in CONFIG["IMAGE_MODEL_CONFIGS"].items()}
    video_model_configs = {name: resolve_model_path(None, path) for name, path in CONFIG["VIDEO_MODEL_CONFIGS"].items()}

    if any(v is None for v in image_model_configs.values()) or any(v is None for v in video_model_configs.values()):
        logging.error("One or more base models for the ensemble were not found. Cannot make predictions."); return
    
    try:
        stacker_model = joblib.load(stacker_model_path)
        image_models = {name: load_model_helper(name, path, args) for name, path in image_model_configs.items()}
        video_models = {name: load_model_helper(name, path, args) for name, path in video_model_configs.items()}
    except Exception as e:
        logging.error(f"Error loading one or more models: {e}"); return

    # 2. Define the data transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Process each input file
    for file_path in args.files_to_predict:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}. Skipping."); continue

        basename, _ = os.path.splitext(file_path)
        
        # Find the corresponding image and video pair
        img_path = next((f for f in [f"{basename}.jpg", f"{basename}.jpeg", f"{basename}.png"] if os.path.exists(f)), None)
        vid_path = next((f for f in [f"{basename}.webm", f"{basename}.mp4", f"{basename}.avi", f"{basename}.mov", f"{basename}.mkv"] if os.path.exists(f)), None)

        if not img_path or not vid_path:
            logging.warning(f"Could not find a matching image/video pair for '{basename}'. Skipping."); continue
            
        try:
            # Prepare image tensor
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Prepare video tensor
            frames = sample_video_frames(vid_path, args.num_frames)
            if not frames:
                logging.warning(f"Could not extract frames from video: {vid_path}. Skipping."); continue
            vid_tensor = torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)

            # Get predictions from all base models
            with torch.no_grad():
                base_predictions = {}
                # NOTE: The class_to_idx logic is hardcoded here to match training (meteor=0).
                # A more robust system might save this mapping, but this is fine for now.
                for name, model in image_models.items():
                    pred = model(img_tensor).item()
                    base_predictions[f"pred_{name}"] = 1.0 - pred
                for name, model in video_models.items():
                    pred = model(vid_tensor).item()
                    base_predictions[f"pred_{name}"] = 1.0 - pred
            
            # Create a DataFrame in the correct order for the stacker model
            feature_names = stacker_model.feature_names_in_
            prediction_df = pd.DataFrame([base_predictions])[feature_names]

            # Get the final probability from the stacker
            final_prob = stacker_model.predict_proba(prediction_df)[0][1]
            
            print(f"{os.path.basename(basename)}: {final_prob:.6f}")

        except Exception as e:
            logging.error(f"Failed to process '{file_path}': {e}")


def build_ensemble_mode(args):
    """Automates the entire process of tuning and training all models."""
    logging.info("--- 🚀 Starting Full Ensemble Build Process 🚀 ---")
    if args.resume:
        logging.info("--- Resuming previous build ---")
    logging.warning("This is a long-running process that will train multiple advanced models.")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_in_tempdir(temp_dir, args)
        
        # --- Train Image Models ---
        for model_name, model_path in CONFIG["IMAGE_MODEL_CONFIGS"].items():
            if args.resume and os.path.exists(model_path):
                logging.info(f"Resuming: Found existing model '{model_path}'. Skipping.")
                continue

            img_args = argparse.Namespace(**vars(args), input_type='image', model_name=model_name)
            _run_training(img_args, data_dir, model_path, params_file=None)
            
        # --- Train Video Models ---
        for model_name, model_path in CONFIG["VIDEO_MODEL_CONFIGS"].items():
            if args.resume and os.path.exists(model_path):
                logging.info(f"Resuming: Found existing model '{model_path}'. Skipping.")
                continue

            vid_args = argparse.Namespace(**vars(args), input_type='video', model_name=model_name)
            _run_training(vid_args, data_dir, model_path, params_file=None)
            
    # --- Build Stacker Model (Train and Tune) ---
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
    """Trains the stacking meta-learner using pre-trained base models."""
    logging.info("--- Training Stacking Meta-Learner ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_in_tempdir(temp_dir, args)
        _, class_to_idx = get_dataloaders(argparse.Namespace(**vars(args), input_type='image'), data_dir, split=False)
        
        image_model_configs = {name: resolve_model_path(None, path) for name, path in CONFIG["IMAGE_MODEL_CONFIGS"].items()}
        video_model_configs = {name: resolve_model_path(None, path) for name, path in CONFIG["VIDEO_MODEL_CONFIGS"].items()}

        if any(v is None for v in image_model_configs.values()) or any(v is None for v in video_model_configs.values()):
            logging.error("One or more base models were not found. Run 'buildensemble' first."); return

        data = _get_stacking_data(args, image_model_configs, video_model_configs, data_dir, class_to_idx)
        data.to_csv(CONFIG['STACKING_DATA_CSV'], index=False)
        logging.info(f"Stacking data created with {len(data)} samples. Training meta-learner...")

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

    # -- Custom Model Modes --
    p_train = subparsers.add_parser('train', help="Train the custom image model.", parents=[parent_parser, dir_parent_parser, balance_parent_parser])
    p_train.add_argument('--params-file', help="Path to a .json file with hyperparameters."); p_train.add_argument('--epochs', type=int, default=50); p_train.add_argument('-o', '--output', default=CONFIG["CUSTOM_IMAGE_MODEL_PATH"])
    
    p_vtrain = subparsers.add_parser('videotrain', help="Train the custom video model.", parents=[parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    p_vtrain.add_argument('--params-file', help="Path to a .json file with hyperparameters."); p_vtrain.add_argument('--epochs', type=int, default=50); p_vtrain.add_argument('-o', '--output', default=CONFIG["CUSTOM_VIDEO_MODEL_PATH"])
    
    p_eval = subparsers.add_parser('evaluate', help="Evaluate the custom image model.", parents=[parent_parser, dir_parent_parser])
    p_eval.add_argument('-m', '--model-file', default=CONFIG["CUSTOM_IMAGE_MODEL_PATH"])
    
    p_veval = subparsers.add_parser('videoevaluate', help="Evaluate the custom video model.", parents=[parent_parser, dir_parent_parser, video_parent_parser])
    p_veval.add_argument('-m', '--model-file', default=CONFIG["CUSTOM_VIDEO_MODEL_PATH"])
    
    p_tune = subparsers.add_parser('tune', help="Tune the custom image model.", parents=[parent_parser, dir_parent_parser, balance_parent_parser])
    p_tune.add_argument('--trials', type=int, default=20); p_tune.add_argument('-o', '--output', default=CONFIG["IMAGE_PARAMS_FILE"])

    p_vtune = subparsers.add_parser('videotune', help="Tune the custom video model.", parents=[parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    p_vtune.add_argument('--trials', type=int, default=20); p_vtune.add_argument('-o', '--output', default=CONFIG["VIDEO_PARAMS_FILE"])

    # -- Stacking Ensemble Modes --
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

    # -- Predict Mode --
    p_predict = subparsers.add_parser('predict', help="Classify files using the stacking ensemble.", parents=[parent_parser, video_parent_parser])
    p_predict.add_argument('files_to_predict', nargs='+', help="Path to image or video file(s) to classify. The script will find matching pairs.")

    args = parser.parse_args()
    setup_logging(args.verbose)
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, running on CPU. This will be very slow.")
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
        # Assign default model name for tuning
        if args.mode == 'tune': args.model_name = 'custom_image'
        if args.mode == 'videotune': args.model_name = 'custom_video'
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = prepare_data_in_tempdir(temp_dir, args)
            _run_tuning(args, data_dir, args.output)
    else:
        # Handle generic train/evaluate legacy modes
        args.input_type = 'video' if 'video' in args.mode else 'image'
        run_generic_mode(args)

if __name__ == '__main__':
    main()
