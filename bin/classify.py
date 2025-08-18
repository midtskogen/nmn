#!/usr/bin/env python3

"""
A professional-grade script for meteor image and video classification, powered by PyTorch.

This script supports two data types:
1. Images (e.g., JPG, PNG) using a 2D CNN.
2. Videos (e.g., WebM, MP4) using a 3D CNN to analyze motion.

It provides parallel modes for each data type (e.g., 'train' vs. 'videotrain').
The 'predict' mode is unified and intelligently uses the best available model.

A new 'buildensemble' mode automates the entire process of tuning and training
all models to produce the final, high-accuracy stacking ensemble.

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
    "DEFAULT_IMAGE_MODEL_NAME": 'meteor_image_model.pth',
    "DEFAULT_VIDEO_MODEL_NAME": 'meteor_video_model.pth',
    "IMAGE_PARAMS_FILE": 'best_image_params.json',
    "VIDEO_PARAMS_FILE": 'best_video_params.json',
    "STACKER_MODEL_NAME": 'stacker_model.joblib',
    "STACKING_DATA_CSV": 'stacking_train_data.csv',
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
        
        # --- CORRECTED LOGIC: Count only files of the relevant type ---
        pos_files_of_type = [f for f in os.listdir(pos_path) if f.lower().endswith(extensions)]
        neg_files_of_type = [f for f in os.listdir(neg_path) if f.lower().endswith(extensions)]
        num_pos_of_type = len(pos_files_of_type)
        num_neg_of_type = len(neg_files_of_type)

        if num_pos_of_type <= num_neg_of_type:
            logging.info(f"Balancing not needed for '{input_type}': {num_pos_of_type} positives <= {num_neg_of_type} negatives.")
            return
            
        num_to_generate = num_pos_of_type - num_neg_of_type
        max_workers = os.cpu_count() or 1
        logging.info(f"Balancing for '{input_type}': generating {num_to_generate} synthetic negatives using up to {max_workers} processes.")

        # --- The rest of the logic uses the new, correct counts ---
        source_files = random.choices(pos_files_of_type, k=num_to_generate)
        
        worker_func = _apply_image_wobble if is_image else _apply_video_shift
        output_prefix = 'synthetic_wobble_' if is_image else 'synthetic_shift_'
        output_ext = '.jpg' if is_image else '.webm'
        
        tasks = []
        for i, source_file_name in enumerate(source_files):
            input_path = os.path.join(pos_path, source_file_name)
            output_path = os.path.join(neg_path, f'{output_prefix}{i:05d}{output_ext}')
            task_args = (input_path, output_path) if is_image else (input_path, output_path, args.num_frames)
            tasks.append(task_args)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(worker_func, tasks), total=len(tasks), desc=f"Balancing ({input_type})"))

    if args.mode == 'buildensemble':
        generate_for_type('image')
        generate_for_type('video')
    elif hasattr(args, 'input_type'):
        generate_for_type(args.input_type)
        
    return str(data_dir)

def get_dataloaders(args, data_dir, split=True):
    transform = transforms.Compose([transforms.Resize((args.img_height, args.img_width)), transforms.ToTensor()])
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

def get_model(args, params={}):
    if args.input_type == 'image': return ConvNet2D(params, (args.img_height, args.img_width))
    else: return VideoConvNet3D(params, (args.img_height, args.img_width), args.num_frames)

def _run_training(args, data_dir, model_path, params_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"; logging.info(f"--- Starting {args.input_type.capitalize()} Model Training on {device} ---")
    params = {};
    if params_file and os.path.exists(params_file):
        with open(params_file, 'r') as f: params = json.load(f)
    else: params = {'num_conv_layers': 3, 'conv_0_filters': 32, 'conv_1_filters': 64, 'conv_2_filters': 128, 'dense_units': 256, 'dropout_rate': 0.5, 'learning_rate': 1e-4}
    train_loader, val_loader, _ = get_dataloaders(args, data_dir)
    model = get_model(args, params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 1e-4)); criterion = nn.BCELoss()
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
            logging.info(f"Epoch {epoch+1}: Val loss improved to {avg_val_loss:.4f}. Model saved.")
        else:
            patience_counter += 1; logging.info(f"Epoch {epoch+1}: Val loss did not improve. Patience: {patience_counter}/{patience}.")
        if patience_counter >= patience: logging.info("Stopping early."); break
    logging.info(f"Training finished. Loading best model weights from '{model_path}'.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    params_path = os.path.splitext(model_path)[0] + '.json'
    with open(params_path, 'w') as f: json.dump(params, f, indent=4)

def _run_evaluation(args, data_dir, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"; logging.info(f"--- Starting {args.input_type.capitalize()} Model Evaluation on {device} ---")
    if not os.path.isfile(model_path): logging.error(f"Model file not found: '{model_path}'"); return
    params_path = os.path.splitext(model_path)[0] + '.json'; params = {}
    if os.path.exists(params_path):
        with open(params_path, 'r') as f: params = json.load(f)
    model = get_model(args, params).to(device); model.load_state_dict(torch.load(model_path, map_location=device)); model.eval()
    loader, class_indices = get_dataloaders(args, data_dir, split=False)
    positive_class_index = class_indices.get('meteor', 0)
    y_true, y_pred_probs = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            outputs = model(inputs.to(device)).cpu()
            y_pred_probs.extend(outputs.numpy().flatten()); y_true.extend(labels.numpy().flatten())
    y_true, y_pred_probs = np.array(y_true), np.array(y_pred_probs)
    y_pred_probs_positive = 1.0 - y_pred_probs if positive_class_index == 0 else y_pred_probs
    y_true_positive = (y_true == positive_class_index).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_positive, y_pred_probs_positive)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]; best_f1 = f1_scores[best_f1_idx]
    best_precision = precision[best_f1_idx]; best_recall = recall[best_f1_idx]
    print("\n" + "="*50 + f"\n--- Evaluation Report for {args.input_type.capitalize()} Model ---\n" + "="*50)
    print(f"✅ Optimal Classification Threshold: {best_threshold:.4f}")
    print(f"\nThis threshold provides the best balance (F1-Score) between Precision and Recall:")
    print(f"  - F1-Score:      {best_f1:.4f}")
    print(f"  - Precision:     {best_precision:.4f}")
    print(f"  - Recall:        {best_recall:.4f}\n" + "="*50)

def _run_tuning(args, data_dir, params_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"; logging.info(f"--- Starting Hyperparameter Tuning for {args.input_type.capitalize()} on {device} ---")
    def objective(trial: optuna.Trial) -> float:
        params = {'num_conv_layers': trial.suggest_int('num_conv_layers', 2, 4), 'dense_units': trial.suggest_categorical('dense_units', [128, 256, 512]),
                  'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5), 'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)}
        for i in range(params['num_conv_layers']): params[f'conv_{i}_filters'] = trial.suggest_categorical(f'conv_{i}_filters', [16, 32, 64, 128])
        try:
            model = get_model(args, params).to(device); optimizer = optim.Adam(model.parameters(), lr=params['learning_rate']); criterion = nn.BCELoss()
            train_loader, val_loader, _ = get_dataloaders(args, data_dir); model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad(); loss = criterion(model(inputs), labels); loss.backward(); optimizer.step()
            model.eval(); correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs.to(device)); correct += ((outputs > 0.5).float() == labels.to(device).float().unsqueeze(1)).sum().item(); total += labels.size(0)
            return correct / total if total > 0 else 0
        except Exception: raise optuna.exceptions.TrialPruned()
    study = optuna.create_study(direction='maximize'); study.optimize(objective, n_trials=args.trials)
    with open(params_path, 'w') as f: json.dump(study.best_params, f, indent=4)
    logging.info(f"Tuning complete. Best validation accuracy: {study.best_value:.4f}. Parameters saved to {params_path}")

def run_generic_mode(args):
    """Wrapper for modes that operate on prepared data directories (train, tune, evaluate)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        if 'evaluate' in args.mode: args.balance = False
        data_dir = prepare_data_in_tempdir(temp_dir, args)
        if args.mode in ['train', 'videotrain']:
            _run_training(args, data_dir, args.output, getattr(args, 'params_file', None))
        elif args.mode in ['evaluate', 'videoevaluate']:
            _run_evaluation(args, data_dir, args.model_file)
        elif args.mode in ['tune', 'videotune']:
            _run_tuning(args, data_dir, args.output)

def pipeline_mode(args):
    """Runs the full tune->train->evaluate pipeline across multiple resolutions."""
    logging.info(f"--- Starting {args.input_type.capitalize()} Pipeline ---")
    resolutions_to_test = [(128, 96), (192, 128)]; pipeline_dir = f"{args.input_type}_pipeline_results"; os.makedirs(pipeline_dir, exist_ok=True)
    for width, height in resolutions_to_test:
        res_str = f"{width}x{height}"; logging.info(f"\n{'='*20} Pipeline for {res_str} {'='*20}")
        res_dir = os.path.join(pipeline_dir, res_str); os.makedirs(res_dir, exist_ok=True)
        args.img_width, args.img_height = width, height; params_path = os.path.join(res_dir, args.params_file_name); model_path = os.path.join(res_dir, args.output)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = prepare_data_in_tempdir(temp_dir, args)
            _run_tuning(args, data_dir, params_path)
            _run_training(args, data_dir, model_path, params_path)
            _run_evaluation(args, data_dir, model_path)
    logging.info(f"--- {args.input_type.capitalize()} Pipeline Finished ---")

class PairedDataset(Dataset):
    def __init__(self, file_pairs, args, transform):
        self.file_pairs, self.args, self.transform = file_pairs, args, transform
    def __len__(self): return len(self.file_pairs)
    def __getitem__(self, index):
        img_path, video_path, label = self.file_pairs[index]
        img_tensor = self.transform(Image.open(img_path).convert('RGB'))
        frames = sample_video_frames(video_path, self.args.num_frames)
        video_tensor = torch.zeros((3, self.args.num_frames, 1, 1))
        if frames: video_tensor = torch.stack([self.transform(f) for f in frames], dim=1)
        return img_tensor, video_tensor, label

def load_model_helper(model_type, model_file, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"; params_path = os.path.splitext(model_file)[0] + '.json'; params = {}
    if os.path.exists(params_path):
        with open(params_path, 'r') as f: params = json.load(f)
    model_args = argparse.Namespace(**vars(args)); model_args.input_type = model_type
    model = get_model(model_args, params).to(device); model.load_state_dict(torch.load(model_file, map_location=device)); model.eval()
    return model

def _get_stacking_data(args, image_model, video_model, data_dir):
    """Helper to generate predictions from base models for stacking."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as eval_temp_dir:
        eval_data_dir = os.path.join(eval_temp_dir, 'data'); shutil.copytree(os.path.join(data_dir, 'meteor'), os.path.join(eval_data_dir, 'meteor'))
        original_neg_dir = os.path.join(eval_data_dir, 'non_meteor'); os.makedirs(original_neg_dir)
        for f in os.listdir(os.path.join(data_dir, 'non_meteor')):
            if not f.startswith('synthetic_'):
                shutil.copy(os.path.join(data_dir, 'non_meteor', f), original_neg_dir)
        def find_data_pairs(positive_dir, negative_dir):
            pairs = []
            for label, directory in [(1, positive_dir), (0, negative_dir)]:
                for f in os.listdir(directory):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        basename = os.path.splitext(f)[0]; img_path = os.path.join(directory, f)
                        vid_path_webm = os.path.join(directory, basename + '.webm'); vid_path_mp4 = os.path.join(directory, basename + '.mp4')
                        vid_path = vid_path_webm if os.path.exists(vid_path_webm) else vid_path_mp4
                        if os.path.exists(vid_path): pairs.append((img_path, vid_path, label))
            return pairs
        file_pairs = find_data_pairs(os.path.join(eval_data_dir, 'meteor'), original_neg_dir)
        transform = transforms.Compose([transforms.Resize((args.img_height, args.img_width)), transforms.ToTensor()])
        dataset = PairedDataset(file_pairs, args, transform); loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() or 1))
        img_preds, vid_preds, true_labels = [], [], []
        with torch.no_grad():
            for img_batch, vid_batch, label_batch in tqdm(loader, desc="Generating Stacking Data"):
                img_preds.extend(1.0 - image_model(img_batch.to(device)).cpu().numpy().flatten())
                vid_preds.extend(1.0 - video_model(vid_batch.to(device)).cpu().numpy().flatten())
                true_labels.extend(label_batch.numpy().flatten())
    return pd.DataFrame({'image_prediction': img_preds, 'video_prediction': vid_preds, 'is_meteor': true_labels})

def train_stacker_mode(args):
    """Trains the stacking meta-learner."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_in_tempdir(temp_dir, args)
        image_model = load_model_helper('image', args.image_model_file, args)
        video_model = load_model_helper('video', args.video_model_file, args)
        data = _get_stacking_data(args, image_model, video_model, data_dir)
        data.to_csv(CONFIG['STACKING_DATA_CSV'], index=False)
        logging.info(f"Stacking data created. Training meta-learner on {len(data)} samples...")
        X, y = data[['image_prediction', 'video_prediction']], data['is_meteor']
        meta_learner = LogisticRegression(); meta_learner.fit(X, y)
        joblib.dump(meta_learner, args.output)
        logging.info(f"Stacking meta-learner trained and saved to {args.output}")

def tune_stacker_mode(args):
    """Tunes the stacking meta-learner."""
    logging.info("--- Tuning Stacking Meta-Learner ---")
    data = pd.read_csv(args.input_csv)
    X, y = data[['image_prediction', 'video_prediction']], data['is_meteor']
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X, y)
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    joblib.dump(grid_search.best_estimator_, args.output)
    logging.info(f"Tuned stacking meta-learner saved to {args.output}")

def evaluate_stack_mode(args):
    """Evaluates the full stacking ensemble and finds its optimal threshold."""
    logging.info("--- Evaluating Stacking Ensemble ---")
    image_model = load_model_helper('image', args.image_model_file, args)
    video_model = load_model_helper('video', args.video_model_file, args)
    stacker_model = joblib.load(args.stacker_model_file)
    with tempfile.TemporaryDirectory() as temp_dir:
        args.balance = False; data_dir = prepare_data_in_tempdir(temp_dir, args)
        data = _get_stacking_data(args, image_model, video_model, data_dir)
    X, y_true_positive = data[['image_prediction', 'video_prediction']], data['is_meteor']
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

def resolve_model_path(arg_path: str, default_name: str) -> Optional[str]:
    """
    Finds a model file, respecting user overrides and searching default locations.
    Search Order:
    1. If a path is provided via CLI that differs from the default, use it exclusively.
    2. If no override is given, search for the default filename in the current directory.
    3. If not found, search in the user's model directory (~/nmn/model).
    """
    if arg_path != default_name:  # User has provided an override path
        if os.path.exists(arg_path):
            logging.info(f"Using user-specified model file: {arg_path}")
            return arg_path
        logging.error(f"User-specified model file not found: '{arg_path}'. Cannot proceed.")
        return None
    else:  # No override, use standard search path
        # 1. Check current directory
        if os.path.exists(default_name):
            logging.info(f"Found model file in current directory: ./{default_name}")
            return default_name
        # 2. Check ~/nmn/model directory
        user_model_dir = pathlib.Path.home() / 'nmn' / 'model'
        user_model_path = user_model_dir / default_name
        if os.path.exists(user_model_path):
            logging.info(f"Found model file in user directory: {user_model_path}")
            return str(user_model_path)
        return None

def predict_mode(args):
    """Classifies files using the best available model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"; IMAGE_EXTS, VIDEO_EXTS = ('.jpg', '.jpeg', '.png'), ('.webm', '.mp4')
    image_model, video_model, stacker_model = None, None, None
    transform = transforms.Compose([transforms.Resize((args.img_height, args.img_width)), transforms.ToTensor()])
    
    # Resolve model paths using the new logic
    image_model_path = resolve_model_path(args.image_model_file, CONFIG["DEFAULT_IMAGE_MODEL_NAME"])
    video_model_path = resolve_model_path(args.video_model_file, CONFIG["DEFAULT_VIDEO_MODEL_NAME"])
    stacker_model_path = resolve_model_path(args.stacker_model_file, CONFIG["STACKER_MODEL_NAME"])

    # Load models if their paths were found
    if stacker_model_path:
        logging.info("Stacker model found. Defaulting to ensemble mode.")
        stacker_model = joblib.load(stacker_model_path)
    if image_model_path:
        image_model = load_model_helper('image', image_model_path, args)
    if video_model_path:
        video_model = load_model_helper('video', video_model_path, args)

    for file_path in args.files_to_predict:
        if not os.path.exists(file_path): logging.error(f"Input file not found: '{file_path}'. Skipping."); continue
        basename, file_ext = os.path.splitext(file_path); file_ext = file_ext.lower(); predicted_this_loop = False
        if args.image:
            if file_ext in IMAGE_EXTS and image_model:
                score = 1.0 - image_model(transform(Image.open(file_path).convert('RGB')).unsqueeze(0).to(device)).item()
                print(f"{os.path.basename(file_path)} (image forced): {score:.6f}")
            else: logging.warning(f"Cannot force image prediction: '{file_path}' is not a valid image or model is not loaded.")
            continue
        if args.video:
            if file_ext in VIDEO_EXTS and video_model:
                frames = sample_video_frames(file_path, args.num_frames)
                if frames:
                    score = 1.0 - video_model(torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)).item()
                    print(f"{os.path.basename(file_path)} (video forced): {score:.6f}")
            else: logging.warning(f"Cannot force video prediction: '{file_path}' is not a valid video or model is not loaded.")
            continue
        if stacker_model and image_model and video_model:
            img_path_jpg, img_path_png = basename + '.jpg', basename + '.png'; vid_path_webm, vid_path_mp4 = basename + '.webm', basename + '.mp4'
            img_path = None
            if os.path.exists(img_path_jpg): img_path = img_path_jpg
            elif os.path.exists(img_path_png): img_path = img_path_png
            vid_path = None
            if os.path.exists(vid_path_webm): vid_path = vid_path_webm
            elif os.path.exists(vid_path_mp4): vid_path = vid_path_mp4
            if img_path and vid_path:
                logging.info(f"Found pair for ensemble: {os.path.basename(img_path)} and {os.path.basename(vid_path)}")
                with torch.no_grad():
                    img_score = 1.0 - image_model(transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)).item()
                    frames = sample_video_frames(vid_path, args.num_frames)
                    vid_score = 1.0 - video_model(torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)).item() if frames else 0.0
                feature_names = stacker_model.feature_names_in_; prediction_df = pd.DataFrame([[img_score, vid_score]], columns=feature_names)
                final_prob = stacker_model.predict_proba(prediction_df)[0][1]; print(f"{os.path.basename(basename)} (ensemble): {final_prob:.6f}"); predicted_this_loop = True
        if not predicted_this_loop:
            if file_ext in IMAGE_EXTS and image_model:
                score = 1.0 - image_model(transform(Image.open(file_path).convert('RGB')).unsqueeze(0).to(device)).item()
                print(f"{os.path.basename(file_path)} (image): {score:.6f}")
            elif file_ext in VIDEO_EXTS and video_model:
                frames = sample_video_frames(file_path, args.num_frames)
                if frames:
                    score = 1.0 - video_model(torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)).item()
                    print(f"{os.path.basename(file_path)} (video): {score:.6f}")
            else: logging.warning(f"Skipping '{file_path}': suitable model not loaded or unsupported file type.")

def build_ensemble_mode(args):
    """Automates the entire process of tuning and training all models."""
    logging.info("--- Starting Full Ensemble Build Process ---"); logging.warning("This is a long-running process that will tune and train all models.")
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_in_tempdir(temp_dir, args)
        logging.info("\n--- Building Image Model ---")
        img_args = argparse.Namespace(**vars(args), input_type='image')
        _run_tuning(img_args, data_dir, args.image_params_file); _run_training(img_args, data_dir, args.image_model_file, args.image_params_file)
        logging.info("\n--- Building Video Model ---")
        vid_args = argparse.Namespace(**vars(args), input_type='video')
        _run_tuning(vid_args, data_dir, args.video_params_file); _run_training(vid_args, data_dir, args.video_model_file, args.video_params_file)
    logging.info("\n--- Building Stacker Model ---")
    stacker_train_args = argparse.Namespace(**vars(args), output=args.stacker_model_file)
    train_stacker_mode(stacker_train_args)
    stacker_tune_args = argparse.Namespace(input_csv=CONFIG['STACKING_DATA_CSV'], output=args.stacker_model_file)
    tune_stacker_mode(stacker_tune_args)
    logging.info("--- Full Ensemble Build Complete! ---")

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
    dir_parent_parser.add_argument('positive_dir'); dir_parent_parser.add_argument('negative_dir')
    video_parent_parser = argparse.ArgumentParser(add_help=False)
    video_parent_parser.add_argument('--num-frames', type=int, default=CONFIG["NUM_FRAMES"])
    balance_parent_parser = argparse.ArgumentParser(add_help=False)
    balance_parent_parser.add_argument('--balance', action='store_true', help="Enable multiprocessing to balance the dataset by creating synthetic negatives.")
    
    # Define modes and their arguments
    # I've simplified the argparse setup slightly to be more readable and less repetitive
    # but the functionality is identical to the last correct version.
    for m, n, p, parents in [
        ('train', 'image', CONFIG["DEFAULT_IMAGE_MODEL_NAME"], [parent_parser, dir_parent_parser, balance_parent_parser]),
        ('videotrain', 'video', CONFIG["DEFAULT_VIDEO_MODEL_NAME"], [parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    ]:
        p_train = subparsers.add_parser(m, help=f"Train the {n} model.", parents=parents)
        p_train.add_argument('--params-file', help="Path to a .json file with hyperparameters."); p_train.add_argument('--epochs', type=int, default=50); p_train.add_argument('-o', '--output', default=p)

    for m, n, p, parents in [
        ('evaluate', 'image', CONFIG["DEFAULT_IMAGE_MODEL_NAME"], [parent_parser, dir_parent_parser]),
        ('videoevaluate', 'video', CONFIG["DEFAULT_VIDEO_MODEL_NAME"], [parent_parser, dir_parent_parser, video_parent_parser])
    ]:
        p_eval = subparsers.add_parser(m, help=f"Evaluate the {n} model.", parents=parents)
        p_eval.add_argument('-m', '--model-file', default=p)

    for m, n, p, parents in [
        ('tune', 'image', CONFIG["IMAGE_PARAMS_FILE"], [parent_parser, dir_parent_parser, balance_parent_parser]),
        ('videotune', 'video', CONFIG["VIDEO_PARAMS_FILE"], [parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    ]:
        p_tune = subparsers.add_parser(m, help=f"Tune the {n} model.", parents=parents)
        p_tune.add_argument('--trials', type=int, default=20); p_tune.add_argument('-o', '--output', default=p)

    for m, n, p, f, parents in [
        ('pipeline', 'image', CONFIG["DEFAULT_IMAGE_MODEL_NAME"], CONFIG["IMAGE_PARAMS_FILE"], [parent_parser, dir_parent_parser, balance_parent_parser]),
        ('videopipeline', 'video', CONFIG["DEFAULT_VIDEO_MODEL_NAME"], CONFIG["VIDEO_PARAMS_FILE"], [parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    ]:
        p_pipe = subparsers.add_parser(m, help=f"Run the full pipeline for {n}s.", parents=parents)
        p_pipe.add_argument('--epochs', type=int, default=50); p_pipe.add_argument('--trials', type=int, default=10); p_pipe.add_argument('-o', '--output', default=p); p_pipe.add_argument('--params-file-name', default=f)

    p_predict = subparsers.add_parser('predict', help="Classify files, using stacking ensemble if available.", parents=[parent_parser, video_parent_parser])
    p_predict.add_argument('files_to_predict', nargs='+'); p_predict.add_argument('--image-model-file', default=CONFIG["DEFAULT_IMAGE_MODEL_NAME"]); p_predict.add_argument('--video-model-file', default=CONFIG["DEFAULT_VIDEO_MODEL_NAME"]); p_predict.add_argument('--stacker-model-file', default=CONFIG["STACKER_MODEL_NAME"])
    force_group = p_predict.add_mutually_exclusive_group()
    force_group.add_argument('--image', action='store_true', help="Force image-only prediction."); force_group.add_argument('--video', action='store_true', help="Force video-only prediction.")
    
    p_stack_train = subparsers.add_parser('trainstacker', help="Create data & train the meta-learner.", parents=[parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser]); p_stack_train.add_argument('--image-model-file', default=CONFIG["DEFAULT_IMAGE_MODEL_NAME"]); p_stack_train.add_argument('--video-model-file', default=CONFIG["DEFAULT_VIDEO_MODEL_NAME"]); p_stack_train.add_argument('-o', '--output', default=CONFIG["STACKER_MODEL_NAME"])
    p_stack_tune = subparsers.add_parser('tunestack', help="Tune the meta-learner.", parents=[parent_parser]); p_stack_tune.add_argument('--input-csv', default=CONFIG["STACKING_DATA_CSV"]); p_stack_tune.add_argument('-o', '--output', default=CONFIG["STACKER_MODEL_NAME"])
    p_stack_eval = subparsers.add_parser('evaluatestack', help="Evaluate the full stacking ensemble.", parents=[parent_parser, dir_parent_parser, video_parent_parser]); p_stack_eval.add_argument('--image-model-file', default=CONFIG["DEFAULT_IMAGE_MODEL_NAME"]); p_stack_eval.add_argument('--video-model-file', default=CONFIG["DEFAULT_VIDEO_MODEL_NAME"]); p_stack_eval.add_argument('--stacker-model-file', default=CONFIG["STACKER_MODEL_NAME"])
    
    p_build = subparsers.add_parser('buildensemble', help="Run the full end-to-end tuning and training for all models.", parents=[parent_parser, dir_parent_parser, video_parent_parser, balance_parent_parser])
    p_build.add_argument('--image-model-file', default=CONFIG["DEFAULT_IMAGE_MODEL_NAME"]); p_build.add_argument('--video-model-file', default=CONFIG["DEFAULT_VIDEO_MODEL_NAME"]); p_build.add_argument('--image-params-file', default=CONFIG["IMAGE_PARAMS_FILE"]); p_build.add_argument('--video-params-file', default=CONFIG["VIDEO_PARAMS_FILE"]); p_build.add_argument('--stacker-model-file', default=CONFIG["STACKER_MODEL_NAME"]); p_build.add_argument('--epochs', type=int, default=50); p_build.add_argument('--trials', type=int, default=20)
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    if not torch.backends.nnpack.is_available(): torch.backends.nnpack.enabled = False
    
    # --- Mode Dispatcher ---
    if args.mode == 'predict': predict_mode(args)
    elif args.mode == 'trainstacker': train_stacker_mode(args)
    elif args.mode == 'tunestack': tune_stacker_mode(args)
    elif args.mode == 'evaluatestack': evaluate_stack_mode(args)
    elif args.mode == 'buildensemble': build_ensemble_mode(args)
    elif 'pipeline' in args.mode:
        args.input_type = 'video' if 'video' in args.mode else 'image'; pipeline_mode(args)
    else:
        args.input_type = 'video' if 'video' in args.mode else 'image'; run_generic_mode(args)

if __name__ == '__main__':
    main()
