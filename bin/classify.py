#!/usr/bin/env python3

"""
A professional-grade script for image and video meteor classification, powered by PyTorch.

This script supports two data types:
1. Images (e.g., JPG, PNG) using a 2D CNN.
2. Videos (e.g., WebM, MP4) using a 3D CNN to analyze motion.

It provides parallel modes for each data type (e.g., 'train' vs. 'videotrain').
The 'predict' mode is unified and intelligently uses the best available model.

A new 'buildensemble' mode automates the entire process of tuning and training
all models to produce the final, high-accuracy stacking ensemble.
"""
# ... (all imports and CONFIG are the same as the previous version) ...
import os
import sys
import argparse
import logging
import json
import pathlib
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple

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

# --- Utility, Models, Data Handling, and Core Logic Functions ---
# (All functions from the previous version like setup_logging, ConvNet2D,
# VideoConvNet3D, sample_video_frames, VideoDataset, get_dataloaders,
# get_model, _run_training, _run_evaluation, _run_tuning are included here,
# unchanged. They are omitted below for brevity but are part of the final script.)

def setup_logging(verbose: bool):
    """Configures logging to be quiet by default and verbose when requested."""
    # If verbose, show INFO messages. Otherwise, only show WARNING and above.
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

def get_dataloaders(args, data_dir, split=True):
    transform = transforms.Compose([transforms.Resize((args.img_height, args.img_width)), transforms.ToTensor()])
    if args.input_type == 'image': full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    else: full_dataset = VideoDataset(data_dir, num_frames=args.num_frames, transform=transform)
    if not hasattr(full_dataset, 'classes') or not full_dataset.classes: raise ValueError(f"No classes found in {data_dir}.")
    num_workers = min(16, os.cpu_count())
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
    params = {}
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
    """A single, shared evaluation loop for both images and videos."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"--- Starting {args.input_type.capitalize()} Model Evaluation on {device} ---")
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
            y_pred_probs.extend(outputs.numpy().flatten())
            y_true.extend(labels.numpy().flatten())

    y_true, y_pred_probs = np.array(y_true), np.array(y_pred_probs)
    y_pred_probs_positive = 1.0 - y_pred_probs if positive_class_index == 0 else y_pred_probs
    y_true_positive = (y_true == positive_class_index).astype(int)
    
    precision, recall, thresholds = precision_recall_curve(y_true_positive, y_pred_probs_positive)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1_idx = np.argmax(f1_scores)
    
    # Extract all the best metrics at the optimal threshold
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    best_precision = precision[best_f1_idx]
    best_recall = recall[best_f1_idx]

    # Display the detailed report
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

# --- Main Mode Wrappers ---
def run_generic_mode(args):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        shutil.copytree(args.positive_dir, os.path.join(data_dir, 'meteor')); shutil.copytree(args.negative_dir, os.path.join(data_dir, 'non_meteor'))
        if args.mode in ['train', 'videotrain']: _run_training(args, data_dir, args.output, getattr(args, 'params_file', None))
        elif args.mode in ['evaluate', 'videoevaluate']: _run_evaluation(args, data_dir, args.model_file)
        elif args.mode in ['tune', 'videotune']: _run_tuning(args, data_dir, args.output)

def pipeline_mode(args):
    logging.info(f"--- Starting {args.input_type.capitalize()} Pipeline ---")
    resolutions_to_test = [(128, 96), (192, 128)]; pipeline_dir = f"{args.input_type}_pipeline_results"; os.makedirs(pipeline_dir, exist_ok=True)
    for width, height in resolutions_to_test:
        res_str = f"{width}x{height}"; logging.info(f"\n{'='*20} Pipeline for {res_str} {'='*20}")
        res_dir = os.path.join(pipeline_dir, res_str); os.makedirs(res_dir, exist_ok=True)
        args.img_width, args.img_height = width, height
        params_path = os.path.join(res_dir, args.params_file_name); model_path = os.path.join(res_dir, args.output)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, 'data')
            shutil.copytree(args.positive_dir, os.path.join(data_dir, 'meteor')); shutil.copytree(args.negative_dir, os.path.join(data_dir, 'non_meteor'))
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params_path = os.path.splitext(model_file)[0] + '.json'; params = {}
    if os.path.exists(params_path):
        with open(params_path, 'r') as f: params = json.load(f)
    model_args = argparse.Namespace(**vars(args)); model_args.input_type = model_type
    model = get_model(model_args, params).to(device); model.load_state_dict(torch.load(model_file, map_location=device)); model.eval()
    return model

def _get_stacking_data(args, image_model, video_model):
    """Helper to generate predictions from base models for stacking."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def find_data_pairs(positive_dir, negative_dir):
        pairs = []
        for label, directory in [(1, positive_dir), (0, negative_dir)]:
            for f in os.listdir(directory):
                if f.lower().endswith('.jpg'):
                    basename = os.path.splitext(f)[0]; img_path = os.path.join(directory, f)
                    vid_path = os.path.join(directory, basename + '.webm')
                    if os.path.exists(vid_path): pairs.append((img_path, vid_path, label))
        return pairs
    
    file_pairs = find_data_pairs(args.positive_dir, args.negative_dir)
    transform = transforms.Compose([transforms.Resize((args.img_height, args.img_width)), transforms.ToTensor()])
    dataset = PairedDataset(file_pairs, args, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count()))
    
    img_preds, vid_preds, true_labels = [], [], []
    with torch.no_grad():
        for img_batch, vid_batch, label_batch in tqdm(loader, desc="Generating Stacking Data"):
            img_preds.extend(1.0 - image_model(img_batch.to(device)).cpu().numpy().flatten())
            vid_preds.extend(1.0 - video_model(vid_batch.to(device)).cpu().numpy().flatten())
            true_labels.extend(label_batch.numpy().flatten())
    return pd.DataFrame({'image_prediction': img_preds, 'video_prediction': vid_preds, 'is_meteor': true_labels})

def train_stacker_mode(args):
    image_model = load_model_helper('image', args.image_model_file, args)
    video_model = load_model_helper('video', args.video_model_file, args)
    data = _get_stacking_data(args, image_model, video_model)
    data.to_csv(CONFIG['STACKING_DATA_CSV'], index=False)
    logging.info(f"Stacking data created. Training meta-learner on {len(data)} samples...")
    X, y = data[['image_prediction', 'video_prediction']], data['is_meteor']
    meta_learner = LogisticRegression(); meta_learner.fit(X, y)
    joblib.dump(meta_learner, args.output)
    logging.info(f"Stacking meta-learner trained and saved to {args.output}")

def tune_stacker_mode(args):
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
    
    data = _get_stacking_data(args, image_model, video_model)
    
    X, y_true_positive = data[['image_prediction', 'video_prediction']], data['is_meteor']
    y_pred_probs_positive = stacker_model.predict_proba(X)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_true_positive, y_pred_probs_positive)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1_idx = np.argmax(f1_scores)

    # Extract all the best metrics at the optimal threshold
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    best_precision = precision[best_f1_idx]
    best_recall = recall[best_f1_idx]
    
    # Display the detailed report
    print("\n" + "="*50 + "\n--- Evaluation Report for Stacking Ensemble ---\n" + "="*50)
    print(f"✅ Optimal Classification Threshold: {best_threshold:.4f}")
    print(f"\nThis threshold provides the best balance (F1-Score) between Precision and Recall:")
    print(f"  - F1-Score:      {best_f1:.4f}")
    print(f"  - Precision:     {best_precision:.4f}")
    print(f"  - Recall:        {best_recall:.4f}\n" + "="*50)

def predict_mode(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"; IMAGE_EXTS, VIDEO_EXTS = ('.jpg', '.jpeg', '.png'), ('.webm', '.mp4')
    image_model, video_model, stacker_model = None, None, None
    transform = transforms.Compose([transforms.Resize((args.img_height, args.img_width)), transforms.ToTensor()])

    if os.path.exists(args.stacker_model_file):
        logging.info("Stacker model found. Defaulting to ensemble mode.")
        stacker_model = joblib.load(args.stacker_model_file)
    
    if os.path.exists(args.image_model_file): image_model = load_model_helper('image', args.image_model_file, args)
    if os.path.exists(args.video_model_file): video_model = load_model_helper('video', args.video_model_file, args)

    for file_path in args.files_to_predict:
        # --- START FIX ---
        # First, check if the provided file path even exists before doing anything else.
        if not os.path.exists(file_path):
            logging.error(f"Input file not found: '{file_path}'. Skipping.")
            continue # Go to the next file in the list
        # --- END FIX ---

        basename, file_ext = os.path.splitext(file_path); file_ext = file_ext.lower()
        predicted_this_loop = False

        # Handle force flags first
        if args.image: # User explicitly requested image prediction
            if file_ext in IMAGE_EXTS and image_model:
                score = 1.0 - image_model(transform(Image.open(file_path).convert('RGB')).unsqueeze(0).to(device)).item()
                print(f"{os.path.basename(file_path)} (image forced): {score:.6f}")
            else:
                logging.warning(f"Cannot force image prediction: '{file_path}' is not a valid image or model is not loaded.")
            continue

        if args.video: # User explicitly requested video prediction
            if file_ext in VIDEO_EXTS and video_model:
                frames = sample_video_frames(file_path, args.num_frames)
                if frames:
                    score = 1.0 - video_model(torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)).item()
                    print(f"{os.path.basename(file_path)} (video forced): {score:.6f}")
            else:
                logging.warning(f"Cannot force video prediction: '{file_path}' is not a valid video or model is not loaded.")
            continue
        
        # Default intelligent logic (if no force flags are used)
        # 1. Attempt to use the stacking ensemble
        if stacker_model and image_model and video_model:
            img_path, vid_path = basename + '.jpg', basename + '.webm'
            if os.path.exists(img_path) and os.path.exists(vid_path):
                logging.info(f"Found pair for ensemble: {os.path.basename(img_path)} and {os.path.basename(vid_path)}")
                with torch.no_grad():
                    img_score = 1.0 - image_model(transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)).item()
                    frames = sample_video_frames(vid_path, args.num_frames)
                    vid_score = 1.0 - video_model(torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)).item() if frames else 0.0
                feature_names = stacker_model.feature_names_in_
                prediction_df = pd.DataFrame([[img_score, vid_score]], columns=feature_names)
                final_prob = stacker_model.predict_proba(prediction_df)[0][1]
                print(f"{os.path.basename(basename)} (ensemble): {final_prob:.6f}")
                predicted_this_loop = True

        # 2. Fall back to single models
        if not predicted_this_loop:
            if file_ext in IMAGE_EXTS and image_model:
                score = 1.0 - image_model(transform(Image.open(file_path).convert('RGB')).unsqueeze(0).to(device)).item()
                print(f"{os.path.basename(file_path)} (image): {score:.6f}")
                predicted_this_loop = True
            elif file_ext in VIDEO_EXTS and video_model:
                frames = sample_video_frames(file_path, args.num_frames)
                if frames:
                    score = 1.0 - video_model(torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)).item()
                    print(f"{os.path.basename(file_path)} (video): {score:.6f}")
                predicted_this_loop = True

        # 3. Final warning if no prediction could be made
        if not predicted_this_loop:
            logging.warning(f"Skipping '{file_path}': suitable model not loaded or unsupported file type.")

def build_ensemble_mode(args):
    """Automates the entire process of tuning and training all models."""
    logging.info("--- Starting Full Ensemble Build Process ---")
    logging.warning("This is a long-running process that will tune and train all models.")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        shutil.copytree(args.positive_dir, os.path.join(data_dir, 'meteor'))
        shutil.copytree(args.negative_dir, os.path.join(data_dir, 'non_meteor'))
        
        # 1. Image Model
        logging.info("\n--- Building Image Model ---")
        img_args = argparse.Namespace(**vars(args))
        img_args.input_type = 'image'
        _run_tuning(img_args, data_dir, args.image_params_file)
        _run_training(img_args, data_dir, args.image_model_file, args.image_params_file)

        # 2. Video Model
        logging.info("\n--- Building Video Model ---")
        vid_args = argparse.Namespace(**vars(args))
        vid_args.input_type = 'video'
        _run_tuning(vid_args, data_dir, args.video_params_file)
        _run_training(vid_args, data_dir, args.video_model_file, args.video_params_file)

    # 3. Stacker Model (uses original data paths)
    logging.info("\n--- Building Stacker Model ---")
    
    # Create a specific set of arguments for the stacker training step
    stacker_train_args = argparse.Namespace(**vars(args))
    stacker_train_args.output = args.stacker_model_file # This is the fix

    train_stacker_mode(stacker_train_args)
    
    # Create a specific set of arguments for the stacker tuning step
    stacker_tune_args = argparse.Namespace(
        input_csv=CONFIG['STACKING_DATA_CSV'], 
        output=args.stacker_model_file
    )
    tune_stacker_mode(stacker_tune_args)

    logging.info("--- Full Ensemble Build Complete! ---")

def main():
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

    # Base Model Modes
    for m, n, p in [('train', 'image', CONFIG["DEFAULT_IMAGE_MODEL_NAME"]), ('videotrain', 'video', CONFIG["DEFAULT_VIDEO_MODEL_NAME"])]:
        p_train = subparsers.add_parser(m, help=f"Train the {n} model.", parents=[parent_parser, dir_parent_parser] + ([video_parent_parser] if n=='video' else []))
        p_train.add_argument('--params-file', help="Path to a .json file with hyperparameters."); p_train.add_argument('--epochs', type=int, default=50); p_train.add_argument('-o', '--output', default=p)
    for m, n, p in [('evaluate', 'image', CONFIG["DEFAULT_IMAGE_MODEL_NAME"]), ('videoevaluate', 'video', CONFIG["DEFAULT_VIDEO_MODEL_NAME"])]:
        p_eval = subparsers.add_parser(m, help=f"Evaluate the {n} model.", parents=[parent_parser, dir_parent_parser] + ([video_parent_parser] if n=='video' else []))
        p_eval.add_argument('-m', '--model-file', default=p)
    for m, n, p in [('tune', 'image', CONFIG["IMAGE_PARAMS_FILE"]), ('videotune', 'video', CONFIG["VIDEO_PARAMS_FILE"])]:
        p_tune = subparsers.add_parser(m, help=f"Tune the {n} model.", parents=[parent_parser, dir_parent_parser] + ([video_parent_parser] if n=='video' else []))
        p_tune.add_argument('--trials', type=int, default=20); p_tune.add_argument('-o', '--output', default=p)
    for m, n, p, f in [('pipeline', 'image', CONFIG["DEFAULT_IMAGE_MODEL_NAME"], CONFIG["IMAGE_PARAMS_FILE"]), ('videopipeline', 'video', CONFIG["DEFAULT_VIDEO_MODEL_NAME"], CONFIG["VIDEO_PARAMS_FILE"])]:
        p_pipe = subparsers.add_parser(m, help=f"Run the full pipeline for {n}s.", parents=[parent_parser, dir_parent_parser] + ([video_parent_parser] if n=='video' else []))
        p_pipe.add_argument('--epochs', type=int, default=50); p_pipe.add_argument('--trials', type=int, default=10); p_pipe.add_argument('-o', '--output', default=p); p_pipe.add_argument('--params-file-name', default=f)

    # Unified Predict Mode
    p_predict = subparsers.add_parser('predict', help="Classify files, using stacking ensemble if available.", parents=[parent_parser, video_parent_parser])
    p_predict.add_argument('files_to_predict', nargs='+'); p_predict.add_argument('--image-model-file', default=CONFIG["DEFAULT_IMAGE_MODEL_NAME"]); p_predict.add_argument('--video-model-file', default=CONFIG["DEFAULT_VIDEO_MODEL_NAME"]); p_predict.add_argument('--stacker-model-file', default=CONFIG["STACKER_MODEL_NAME"])
    force_group = p_predict.add_mutually_exclusive_group()
    force_group.add_argument('--image', action='store_true', help="Force image-only prediction, even if a pair exists.")
    force_group.add_argument('--video', action='store_true', help="Force video-only prediction, even if a pair exists.")

    # Stacking Modes
    p_stack_train = subparsers.add_parser('trainstacker', help="Create data & train the meta-learner.", parents=[parent_parser, dir_parent_parser, video_parent_parser])
    p_stack_train.add_argument('--image-model-file', default=CONFIG["DEFAULT_IMAGE_MODEL_NAME"]); p_stack_train.add_argument('--video-model-file', default=CONFIG["DEFAULT_VIDEO_MODEL_NAME"]); p_stack_train.add_argument('-o', '--output', default=CONFIG["STACKER_MODEL_NAME"])
    p_stack_tune = subparsers.add_parser('tunestack', help="Tune the meta-learner.", parents=[parent_parser])
    p_stack_tune.add_argument('--input-csv', default=CONFIG["STACKING_DATA_CSV"]); p_stack_tune.add_argument('-o', '--output', default=CONFIG["STACKER_MODEL_NAME"])
    p_stack_eval = subparsers.add_parser('evaluatestack', help="Evaluate the full stacking ensemble.", parents=[parent_parser, dir_parent_parser, video_parent_parser])
    p_stack_eval.add_argument('--image-model-file', default=CONFIG["DEFAULT_IMAGE_MODEL_NAME"]); p_stack_eval.add_argument('--video-model-file', default=CONFIG["DEFAULT_VIDEO_MODEL_NAME"]); p_stack_eval.add_argument('--stacker-model-file', default=CONFIG["STACKER_MODEL_NAME"])
    
    # All-in-one Build Mode
    p_build = subparsers.add_parser('buildensemble', help="Run the full end-to-end tuning and training for all models.", parents=[parent_parser, dir_parent_parser, video_parent_parser])
    p_build.add_argument('--image-model-file', default=CONFIG["DEFAULT_IMAGE_MODEL_NAME"]); p_build.add_argument('--video-model-file', default=CONFIG["DEFAULT_VIDEO_MODEL_NAME"])
    p_build.add_argument('--image-params-file', default=CONFIG["IMAGE_PARAMS_FILE"]); p_build.add_argument('--video-params-file', default=CONFIG["VIDEO_PARAMS_FILE"])
    p_build.add_argument('--stacker-model-file', default=CONFIG["STACKER_MODEL_NAME"]); p_build.add_argument('--epochs', type=int, default=50); p_build.add_argument('--trials', type=int, default=20)

    args = parser.parse_args()
    setup_logging(args.verbose)
    if not torch.backends.nnpack.is_available(): torch.backends.nnpack.enabled = False
    
    # Mode Dispatcher
    if args.mode == 'predict': predict_mode(args)
    elif args.mode == 'trainstacker': train_stacker_mode(args)
    elif args.mode == 'tunestack': tune_stacker_mode(args)
    elif args.mode == 'evaluatestack': evaluate_stack_mode(args)
    elif args.mode == 'buildensemble': build_ensemble_mode(args)
    elif 'pipeline' in args.mode:
        args.input_type = 'video' if 'video' in args.mode else 'image'
        pipeline_mode(args)
    else:
        args.input_type = 'video' if 'video' in args.mode else 'image'
        run_generic_mode(args)

if __name__ == '__main__':
    main()
