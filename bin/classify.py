#!/usr/bin/env python3
"""
A professional-grade script for meteor image classification with five modes, powered by PyTorch.
1.  pipeline: Runs the full tune -> train -> evaluate pipeline across multiple resolutions.
2.  tune:     Uses Optuna to find the best model architecture for a single resolution.
3.  train:    Trains a model using either default or tuned hyperparameters.
4.  evaluate: Analyzes a trained model to find the optimal classification threshold.
5.  predict:  Uses a trained model to classify one or more images.

This script includes optional, on-the-fly dataset balancing for training and tuning.
If the --balance flag is used and the number of negative images is less than positive,
it will generate synthetic "wobbly" negatives from the positive images to balance the classes.
Evaluation is always performed on the original, unaltered data.
"""
import os
import sys
import argparse
import logging
import json
import pathlib
import tempfile
import shutil
import contextlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import random
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        logging.info("tqdm not found. To see progress bars, please run 'pip install tqdm'")
        return iterable

# --- PyTorch and related imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Default Configuration ---
CONFIG = {
    "DEFAULT_IMG_HEIGHT": 96,
    "DEFAULT_IMG_WIDTH": 192,
    "DEFAULT_MODEL_NAME": 'meteor_model.pth',
    "PARAMS_FILE": 'best_params.json',
    "INDEX_FILE": 'class_indices.json'
}

# --- Utility Functions (Largely Unchanged) ---

def setup_logging(verbose: bool):
    """Configures the logging module based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@contextlib.contextmanager
def suppress_stderr():
    """A robust context manager to suppress C-level stderr output."""
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        if 'devnull_fd' in locals(): os.close(devnull_fd)

def _get_meteor_index(cli_index: Optional[int], model_path: str) -> int:
    """
    Determines the meteor class index with tiered logic:
    1. Use the command-line argument if provided.
    2. Fall back to the 'class_indices.json' file next to the model.
    3. Default to 0 if neither is found.
    """
    if cli_index is not None:
        logging.info(f"Using user-provided command-line index: {cli_index}")
        return cli_index

    index_file_path = os.path.join(os.path.dirname(model_path), CONFIG["INDEX_FILE"])
    if os.path.exists(index_file_path):
        try:
            with open(index_file_path, 'r') as f:
                indices = json.load(f)
                meteor_index = int(indices.get('meteor', 0))
                logging.info(f"Found and using index from '{index_file_path}': {meteor_index}")
                return meteor_index
        except (ValueError, TypeError, json.JSONDecodeError):
            logging.warning(f"Could not parse index from '{index_file_path}'.")

    logging.warning(f"No index provided and '{index_file_path}' not found. Defaulting to 0.")
    logging.warning("This may lead to incorrect predictions if 'meteor' was not class 0 during training.")
    return 0

def _apply_wobble(input_path: str, output_path: str, amplitude: float, frequency: float):
    """Internal function to apply a vertical wobble effect to an image."""
    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            pixels = np.array(img)
            height, width, _ = pixels.shape
            new_pixels = np.zeros_like(pixels)
            for x in range(width):
                shift = int(amplitude * np.sin(2 * np.pi * x * frequency / width))
                rolled_column = np.roll(pixels[:, x, :], shift, axis=0)
                if shift > 0: rolled_column[:shift, :] = 0
                elif shift < 0: rolled_column[shift:, :] = 0
                new_pixels[:, x, :] = rolled_column
            Image.fromarray(new_pixels).save(output_path)
    except Exception as e:
        logging.warning(f"Could not apply wobble to {input_path}: {e}")

def prepare_data_directories(positive_dir: str, negative_dir: str, temp_dir_path: str, balance_dataset: bool = False) -> str:
    """Copies images into a temporary, structured directory and optionally balances the dataset."""
    data_dir = pathlib.Path(temp_dir_path) / 'data'
    pos_path = data_dir / 'meteor'
    neg_path = data_dir / 'non_meteor'
    pos_path.mkdir(parents=True, exist_ok=True)
    neg_path.mkdir(parents=True, exist_ok=True)

    logging.info("Organizing images into temporary directory...")
    pos_files = [f for f in os.listdir(positive_dir) if os.path.isfile(os.path.join(positive_dir, f))]
    neg_files = [f for f in os.listdir(negative_dir) if os.path.isfile(os.path.join(negative_dir, f))]

    for img_file in pos_files: shutil.copy(os.path.join(positive_dir, img_file), pos_path)
    for img_file in neg_files: shutil.copy(os.path.join(negative_dir, img_file), neg_path)

    if balance_dataset:
        num_pos = len(pos_files)
        num_neg = len(neg_files)
        if num_pos > num_neg:
            num_to_generate = num_pos - num_neg
            logging.info(f"Balancing dataset: {num_pos} positives, {num_neg} negatives. Generating {num_to_generate} wobbly negatives from positive images.")
            source_images_to_wobble = random.choices(pos_files, k=num_to_generate)
            for i, source_img_name in enumerate(tqdm(source_images_to_wobble, desc="Balancing dataset", unit="image", file=sys.stdout, disable=logging.getLogger().level > logging.INFO)):
                input_path = os.path.join(positive_dir, source_img_name)
                output_path = os.path.join(neg_path, f'synthetic_wobble_{i:05d}.jpg')
                amplitude = random.uniform(5.0, 15.0)
                frequency = random.uniform(1.0, 4.0)
                _apply_wobble(input_path, output_path, amplitude, frequency)
    else:
        logging.info("Skipping dataset balancing.")

    return str(data_dir)

# --- PyTorch Model Definition ---

class ConvNet(nn.Module):
    def __init__(self, params: Dict[str, Any], image_size: Tuple[int, int]):
        super(ConvNet, self).__init__()
        img_height, img_width = image_size
        self.layers = nn.ModuleList()

        # Input layer
        in_channels = 3
        
        # Convolutional layers from params
        for i in range(params.get('num_conv_layers', 3)):
            out_channels = params.get(f'conv_{i}_filters', 64)
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        self.layers.append(nn.Flatten())

        # Calculate flattened size to connect to dense layers
        # Use a dummy tensor to probe the output size of the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_height, img_width)
            probe = nn.Sequential(*self.layers)(dummy_input)
            flattened_size = probe.shape[1]

        # Dense layers
        self.layers.append(nn.Linear(flattened_size, params.get('dense_units', 128)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=params.get('dropout_rate', 0.5)))
        self.layers.append(nn.Linear(params.get('dense_units', 128), 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

# --- Core Logic Functions (PyTorch Version) ---

def _get_dataloaders(data_dir: str, image_size: Tuple[int, int], batch_size: int, for_training: bool) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Creates PyTorch DataLoaders."""
    img_height, img_width = image_size
    
    # Augmentation for training, simple resizing for validation
    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    
    transform = train_transform if for_training else val_transform
    
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Split dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # We need to apply the correct transform to the validation subset if we are in training mode
    if for_training:
        val_dataset.dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    class_indices = full_dataset.class_to_idx
    return train_loader, val_loader, class_indices

def tune_model(args: argparse.Namespace):
    """Public-facing wrapper for the tuning logic using Optuna."""
    params_path = os.path.join(os.getcwd(), CONFIG["PARAMS_FILE"])
    _run_tuning(args.positive_dir, args.negative_dir, (args.img_height, args.img_width),
                args.batch_size, args.epochs, os.getcwd(), params_path, args.balance)

    script_name = os.path.basename(sys.argv[0])
    print("\n" + "="*20 + " Next Step " + "="*20)
    print(f"Tuning complete. Best parameters saved to '{params_path}'")
    print("\nTo train a model with these optimal parameters, run:")
    print(f"  python {script_name} train {args.positive_dir} {args.negative_dir} --params-file {params_path} "
          f"--img-width {args.img_width} --img-height {args.img_height}")

def _run_tuning(positive_dir, negative_dir, image_size, batch_size, epochs, output_dir, params_path, balance_dataset):
    import optuna
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_height, img_width = image_size
    logging.info(f"--- Starting Hyperparameter Tuning for resolution {img_width}x{img_height} on {device} ---")

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_directories(positive_dir, negative_dir, temp_dir, balance_dataset=balance_dataset)
        train_loader, val_loader, _ = _get_dataloaders(data_dir, image_size, batch_size, for_training=True)

        def objective(trial: optuna.Trial) -> float:
            params = {
                'num_conv_layers': trial.suggest_int('num_conv_layers', 2, 4),
                'dense_units': trial.suggest_categorical('dense_units', [128, 256, 512]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5, step=0.1),
                'learning_rate': trial.suggest_categorical('learning_rate', [1e-3, 1e-4])
            }
            for i in range(params['num_conv_layers']):
                params[f'conv_{i}_filters'] = trial.suggest_categorical(f'conv_{i}_filters', [32, 64, 128])

            model = ConvNet(params, image_size).to(device)
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.BCELoss()
            
            best_val_accuracy = 0.0
            patience_counter = 0

            for epoch in range(epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss, correct, total = 0, 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()
                        predicted = (outputs > 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_accuracy = correct / total
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                trial.report(val_accuracy, epoch)
                if trial.should_prune() or patience_counter >= 5:
                    raise optuna.exceptions.TrialPruned()
            
            return best_val_accuracy

        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=50, timeout=1800) # 50 trials or 30 minutes
        
        best_hps = study.best_trial.params
        logging.info(f"\n--- Tuning Complete for {img_width}x{img_height} ---\n" +
                     f"Best Validation Accuracy: {study.best_value:.4f}\n" +
                     f"Optimal Parameters: {json.dumps(best_hps, indent=4)}")

        with open(params_path, 'w') as f:
            json.dump(best_hps, f, indent=4)
        logging.info(f"Best parameters saved to {params_path}")
    return params_path

def train_model(args: argparse.Namespace):
    """Public-facing wrapper for the training logic."""
    model_path = os.path.join(os.getcwd(), args.output)
    _run_training(args.positive_dir, args.negative_dir, (args.img_height, args.img_width),
                  args.batch_size, args.epochs, args.params_file, model_path, args.balance)

def _run_training(positive_dir, negative_dir, image_size, batch_size, epochs, params_file, model_path, balance_dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_height, img_width = image_size
    logging.info(f"--- Starting Model Training for resolution {img_width}x{img_height} on {device} ---")
    
    params = {}
    if params_file and os.path.exists(params_file):
        logging.info(f"Loading tuned hyperparameters from {params_file}")
        with open(params_file, 'r') as f: params = json.load(f)
    else:
        logging.warning("No params file found. Using default hyperparameters.")
        params = {'num_conv_layers': 3, 'conv_0_filters': 32, 'conv_1_filters': 64, 
                  'conv_2_filters': 128, 'dense_units': 128, 'dropout_rate': 0.5, 'learning_rate': 1e-3}

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_directories(positive_dir, negative_dir, temp_dir, balance_dataset=balance_dataset)
        train_loader, val_loader, class_indices = _get_dataloaders(data_dir, image_size, batch_size, for_training=True)
        
        logging.info(f"Classes found: {class_indices}")
        meteor_class_index = class_indices.get('meteor', 0)

        model = ConvNet(params, image_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 1e-3))
        criterion = nn.BCELoss()
        
        logging.info("Model Summary:")
        logging.info(model)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 8

        logging.info("Starting training with automated stopping...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False, file=sys.stdout):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # Early stopping and model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_path)
                logging.info(f"Validation loss decreased. Saving model to {model_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Validation loss did not improve for {patience} epochs. Stopping early.")
                    break
        
    index_file_path = os.path.join(os.path.dirname(model_path), CONFIG["INDEX_FILE"])
    with open(index_file_path, 'w') as f:
        json.dump(class_indices, f, indent=4)

    logging.info(f"Training complete. Best model saved to '{model_path}'")
    logging.info(f"Class indices saved to '{index_file_path}'")
    return model_path

def evaluate_model(args: argparse.Namespace):
    """Public-facing wrapper for the evaluation logic."""
    _run_evaluation(args.positive_dir, args.negative_dir, (args.img_height, args.img_width),
                    args.batch_size, args.model_file, cli_meteor_index=args.meteor_class_index)

def _run_evaluation(positive_dir, negative_dir, image_size, batch_size, model_path, cli_meteor_index: Optional[int] = None):
    from sklearn.metrics import precision_recall_curve, confusion_matrix
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_height, img_width = image_size
    logging.info(f"--- Starting Model Evaluation for {img_width}x{img_height} on {device} ---")
    if not os.path.isfile(model_path):
        logging.error(f"Model file not found at '{model_path}'"); return None

    # Load dummy model to infer architecture from params file, then load state_dict
    params = {}
    params_path = os.path.join(os.path.dirname(model_path), CONFIG['PARAMS_FILE'])
    if os.path.exists(params_path):
        with open(params_path, 'r') as f: params = json.load(f)
    else: # Default params if none found
        params = {'num_conv_layers': 3, 'conv_0_filters': 32, 'conv_1_filters': 64, 
                  'conv_2_filters': 128, 'dense_units': 128, 'dropout_rate': 0.5}

    model = ConvNet(params, image_size).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logging.error(f"Failed to load model state: {e}. Ensure params file matches model architecture.")
        return None

    model.eval()
    
    meteor_class_index = _get_meteor_index(cli_meteor_index, model_path)

    results = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_directories(positive_dir, negative_dir, temp_dir, balance_dataset=False)
        _, val_loader, class_indices = _get_dataloaders(data_dir, image_size, batch_size, for_training=False)

        y_true, y_pred_probs = [], []
        logging.info("Extracting true labels and predicting probabilities...")
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Evaluating", unit="batch", file=sys.stdout):
                inputs = inputs.to(device)
                outputs = model(inputs).cpu()
                y_pred_probs.extend(outputs.numpy().flatten())
                y_true.extend(labels.numpy().flatten())

        y_true, y_pred_probs = np.array(y_true), np.array(y_pred_probs)
        
        # In PyTorch ImageFolder, class names are sorted alphabetically.
        # 'meteor' is class 0, 'non_meteor' is class 1.
        # The model predicts the probability of class 1 (non_meteor).
        # We want the probability of meteor (class 0).
        y_pred_probs_meteor = 1 - y_pred_probs 
        # True labels are also 0 for meteor, 1 for non_meteor. We need to flip them for sklearn.
        y_true_meteor = 1 - y_true 

        logging.info("Calculating metrics across all thresholds...")
        precision, recall, thresholds = precision_recall_curve(y_true_meteor, y_pred_probs_meteor)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        best_f1, best_precision, best_recall = f1_scores[best_f1_idx], precision[best_f1_idx], recall[best_f1_idx]
        y_pred_binary = (y_pred_probs_meteor >= best_threshold).astype(int)
        
        results = {
            "resolution": f"{img_width}x{img_height}", "f1_score": float(best_f1),
            "precision": float(best_precision), "recall": float(best_recall), "threshold": float(best_threshold)}
        logging.info(f"\n--- Evaluation Report for {img_width}x{img_height} ---")
        print(f"Optimal Threshold: {results['threshold']:.4f}")
        print(f"  - F1-Score:      {results['f1_score']:.4f}")
        print(f"  - Precision:     {results['precision']:.4f}")
        print(f"  - Recall:        {results['recall']:.4f}")
    return results

def predict_image(args: argparse.Namespace):
    """Classifies one or more images using a trained PyTorch model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isfile(args.model_file):
        logging.error(f"Model file not found at '{args.model_file}'"); sys.exit(1)

    logging.info(f"Loading model: {args.model_file}")
    
    # We don't know the image size or architecture, so we must load params
    params = {}
    params_path = os.path.join(os.path.dirname(args.model_file), CONFIG['PARAMS_FILE'])
    if not os.path.exists(params_path):
        logging.error(f"Cannot predict: Parameters file '{params_path}' not found alongside the model.")
        sys.exit(1)
        
    with open(params_path, 'r') as f: params = json.load(f)
    
    # A bit of a chicken-and-egg problem. We need image size to build the model,
    # but that's not stored in the params file. We assume the user has set it correctly
    # or is using a pipeline-generated model where we could store it.
    # For now, we will rely on command-line args. This is a weakness. A better
    # approach would be to save metadata with the model.
    # Let's assume the model was trained with the default dimensions if not specified.
    img_height = args.img_height if 'img_height' in args else CONFIG["DEFAULT_IMG_HEIGHT"]
    img_width = args.img_width if 'img_width' in args else CONFIG["DEFAULT_IMG_WIDTH"]

    model = ConvNet(params, (img_height, img_width)).to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()

    meteor_class_index = _get_meteor_index(args.meteor_class_index, args.model_file)

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    for image_path in args.image_files:
        if not os.path.isfile(image_path):
            logging.warning(f"Skipping '{image_path}': file not found."); continue
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(img_tensor).item() # returns prob of non-meteor

            meteor_probability = 1.0 - prediction
            print(f"{os.path.basename(image_path)}: {meteor_probability:.6f}")
        except Exception as e:
            logging.error(f"Failed to process {image_path}: {e}")

def run_pipeline(args: argparse.Namespace):
    """Runs the full tune->train->evaluate pipeline across multiple resolutions."""
    # This function's logic remains largely the same, only the calls to the
    # underlying _run_* functions are different.
    heights = [64, 96, 128]
    widths = [96, 128, 160, 192, 224, 256]
    resolutions_to_test = sorted([(w, h) for w in widths for h in heights if w >= h])
    
    pipeline_dir = "pipeline_results"
    summary_file_path = os.path.join(pipeline_dir, "summary.json")
    
    all_results = []
    completed_resolutions = set()

    is_fresh_run = not args.resume and not args.preliminary

    if is_fresh_run:
        if os.path.exists(pipeline_dir):
            logging.info(f"Starting new pipeline. Removing existing results in '{pipeline_dir}'...")
            shutil.rmtree(pipeline_dir)
        os.makedirs(pipeline_dir, exist_ok=True)
        logging.info(f"Starting new pipeline. Results will be stored in '{pipeline_dir}'")
    else:
        # Load existing results for a resume or a preliminary report.
        os.makedirs(pipeline_dir, exist_ok=True)
        logging.info(f"Attempting to load existing results from '{pipeline_dir}'...")
        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r') as f:
                    all_results = json.load(f)
                for result in all_results:
                    completed_resolutions.add(result['resolution'])
                if completed_resolutions:
                    logging.info(f"Loaded {len(completed_resolutions)} completed resolutions from summary.json.")
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not read summary file at '{summary_file_path}'. Will check for individual result files. Error: {e}")
                all_results = []
        
        logging.info("Checking for individually completed runs to recover...")
        for width, height in resolutions_to_test:
            res_str = f"{width}x{height}"
            if res_str in completed_resolutions:
                continue

            res_dir = os.path.join(pipeline_dir, res_str)
            model_path = os.path.join(res_dir, CONFIG['DEFAULT_MODEL_NAME'])
            
            if os.path.exists(model_path):
                logging.info(f"Found existing model for {res_str}. Re-running evaluation to recover results.")
                try:
                    eval_results = _run_evaluation(args.positive_dir, args.negative_dir, (height, width),
                                                   args.batch_size, model_path, cli_meteor_index=None)
                    if eval_results:
                        all_results.append(eval_results)
                        completed_resolutions.add(res_str)
                        logging.info(f"Successfully recovered results for {res_str}.")
                except Exception as e:
                    logging.error(f"Failed to recover results for {res_str} due to an error during evaluation: {e}")
            
        if all_results:
            with open(summary_file_path, 'w') as f:
                json.dump(all_results, f, indent=4)
    
    if args.preliminary and not args.resume:
        if not all_results:
            logging.warning(f"No results found in '{pipeline_dir}' to report on.")
            return

        all_results.sort(key=lambda x: x['f1_score'], reverse=True)
        best_result = all_results[0]
        
        logging.info("\n" + "="*50 + "\n--- Pipeline Status Report ---\n" + "="*50)
        print(f"Best overall performance found for resolution {best_result['resolution']}\n")
        print(f"| {'Resolution':<12} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Optimal Threshold':<20} |")
        print(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*22}|")
        for result in all_results:
            print(f"| {result['resolution']:<12} | {result['f1_score']:.4f}     | {result['precision']:.4f}     | {result['recall']:.4f}   | {result['threshold']:.4f}               |")
        
        best_res_dir = os.path.join(pipeline_dir, best_result['resolution'])
        print("\n" + "="*20 + " Best Model Location " + "="*20)
        print(f"The best model and its parameters are located in:")
        print(f"  Model:       {os.path.join(best_res_dir, CONFIG['DEFAULT_MODEL_NAME'])}")
        print(f"  Parameters:  {os.path.join(best_res_dir, CONFIG['PARAMS_FILE'])}")
        print("\nTo continue the pipeline run, use the --resume flag.")
        return

    for width, height in resolutions_to_test:
        res_str = f"{width}x{height}"
        if res_str in completed_resolutions:
            logging.info(f"Skipping already completed resolution: {res_str}")
            continue

        res_dir = os.path.join(pipeline_dir, res_str)
        os.makedirs(res_dir, exist_ok=True)
        
        params_path = os.path.join(res_dir, CONFIG['PARAMS_FILE'])
        model_path = os.path.join(res_dir, CONFIG['DEFAULT_MODEL_NAME'])
        
        try:
            logging.info(f"\n{'='*20} Starting pipeline for {res_str} {'='*20}")
            _run_tuning(args.positive_dir, args.negative_dir, (height, width),
                        args.batch_size, args.tune_epochs, res_dir, params_path, args.balance)
            _run_training(args.positive_dir, args.negative_dir, (height, width),
                          args.batch_size, args.train_epochs, params_path, model_path, args.balance)
            eval_results = _run_evaluation(args.positive_dir, args.negative_dir, (height, width),
                                           args.batch_size, model_path, cli_meteor_index=None)
            if eval_results:
                all_results.append(eval_results)
                with open(summary_file_path, 'w') as f:
                    json.dump(all_results, f, indent=4)
                logging.info(f"Successfully completed and saved results for {res_str}")

                if args.preliminary:
                    temp_results = sorted(all_results, key=lambda x: x['f1_score'], reverse=True)
                    best_prelim_result = temp_results[0]
                    
                    logging.info("\n" + "="*50 + "\n--- Preliminary Pipeline Report (Tuning ongoing) ---\n" + "="*50)
                    print(f"Best performance so far for resolution {best_prelim_result['resolution']}\n")
                    print(f"| {'Resolution':<12} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Optimal Threshold':<20} |")
                    print(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*22}|")
                    for result in temp_results:
                        print(f"| {result['resolution']:<12} | {result['f1_score']:.4f}     | {result['precision']:.4f}     | {result['recall']:.4f}   | {result['threshold']:.4f}               |")
                    print("\n" + "="*20 + " Continuing to next resolution " + "="*20)

        except Exception as e:
            logging.error(f"Pipeline failed for resolution {width}x{height}: {e}", exc_info=args.verbose)
            logging.info("Continuing to next resolution...")
            continue

    if not all_results:
        logging.error("Pipeline finished but no results were generated. Please check logs for errors.")
        return

    all_results.sort(key=lambda x: x['f1_score'], reverse=True)
    best_result = all_results[0]
    
    logging.info("\n" + "="*50 + "\n--- Pipeline Complete: Final Summary ---\n" + "="*50)
    print(f"Best overall performance found for resolution {best_result['resolution']}\n")
    print(f"| {'Resolution':<12} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Optimal Threshold':<20} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*22}|")
    for result in all_results:
        print(f"| {result['resolution']:<12} | {result['f1_score']:.4f}     | {result['precision']:.4f}     | {result['recall']:.4f}   | {result['threshold']:.4f}               |")
    
    best_res_dir = os.path.join(pipeline_dir, best_result['resolution'])
    print("\n" + "="*20 + " Best Model Location " + "="*20)
    print(f"The best model and its parameters are located in:")
    print(f"  Model:       {os.path.join(best_res_dir, CONFIG['DEFAULT_MODEL_NAME'])}")
    print(f"  Parameters:  {os.path.join(best_res_dir, CONFIG['PARAMS_FILE'])}")

def main():
    # --- Argument Parsing (Largely Unchanged) ---
    parser = argparse.ArgumentParser(description="A tool for meteor classification using PyTorch.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose logging output.")
    
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Operating mode")
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--img-width', type=int, default=CONFIG["DEFAULT_IMG_WIDTH"], help="Image width for processing.")
    parent_parser.add_argument('--img-height', type=int, default=CONFIG["DEFAULT_IMG_HEIGHT"], help="Image height for processing.")
    parent_parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training/evaluation.")
    
    balance_parent_parser = argparse.ArgumentParser(add_help=False)
    balance_parent_parser.add_argument('--balance', action='store_true', help="Enable on-the-fly dataset balancing.")

    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('positive_dir', help="Directory with positive meteor images.")
    dir_parent_parser.add_argument('negative_dir', help="Directory with negative (non-meteor) images.")

    meteor_index_parser = argparse.ArgumentParser(add_help=False)
    meteor_index_parser.add_argument('--meteor-class-index', type=int, default=None, choices=[0, 1],
                                     help="Override the class index for 'meteor'. If not provided, it's inferred\n"
                                          "from the '" + CONFIG["INDEX_FILE"] + "' file or defaults to 0.")

    parser_pipeline = subparsers.add_parser('pipeline', help="Run the full tune->train->evaluate workflow.",
                                            parents=[dir_parent_parser, balance_parent_parser],
                                            formatter_class=argparse.RawTextHelpFormatter,
                                            epilog="example:\n  python %(prog)s ./pos/ ./neg/ --balance --resume")
    parser_pipeline.add_argument('--tune-epochs', type=int, default=20, help="Max epochs for each tuning trial.")
    parser_pipeline.add_argument('--train-epochs', type=int, default=100, help="Max epochs for the final training stage.")
    parser_pipeline.add_argument('--batch-size', type=int, default=32, help="Batch size used throughout the pipeline.")
    parser_pipeline.add_argument('--resume', action='store_true', help="Resume a previously stopped pipeline run.")
    parser_pipeline.add_argument('--preliminary', action='store_true', help="Print a report of existing results and exit. If used with --resume, prints intermediate reports during the run.")

    parser_tune = subparsers.add_parser('tune', help="Find the best model hyperparameters.", parents=[parent_parser, balance_parent_parser, dir_parent_parser],
                                        formatter_class=argparse.RawTextHelpFormatter,
                                        epilog="example:\n  python %(prog)s ./pos/ ./neg/ --balance --img-width 192")
    parser_tune.add_argument('--epochs', type=int, default=20, help="Max epochs per tuning trial.")
    
    parser_train = subparsers.add_parser('train', help="Train the model.", parents=[parent_parser, balance_parent_parser, dir_parent_parser],
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         epilog="""examples:\n  python %(prog)s ./pos/ ./neg/ --balance -o v2.pth""")
    parser_train.add_argument('--params-file', default=CONFIG["PARAMS_FILE"], help="JSON file with tuned hyperparameters.")
    parser_train.add_argument('--epochs', type=int, default=100, help="Maximum number of training epochs.")
    parser_train.add_argument('-o', '--output', default=CONFIG["DEFAULT_MODEL_NAME"], help="Output filename for the model.")
    
    parser_evaluate = subparsers.add_parser('evaluate', help="Find the optimal classification threshold.", parents=[parent_parser, dir_parent_parser, meteor_index_parser],
                                            formatter_class=argparse.RawTextHelpFormatter,
                                            epilog="""example:\n  python %(prog)s ./pos/ ./neg/ -m model.pth\n  python %(prog)s ./pos/ ./neg/ -m m.pth --meteor-class-index 0""")
    parser_evaluate.add_argument('-m', '--model-file', default=CONFIG["DEFAULT_MODEL_NAME"], help="Path to the trained model file.")
    
    parser_predict = subparsers.add_parser('predict', help="Classify one or more images.", parents=[meteor_index_parser, parent_parser], formatter_class=argparse.RawTextHelpFormatter,
                                           epilog="""examples:\n  python %(prog)s image1.jpg\n  python %(prog)s i.jpg -m m.pth --meteor-class-index 0""")
    parser_predict.add_argument('image_files', nargs='+', help="One or more image file paths to classify.")
    parser_predict.add_argument('-m', '--model-file', default=CONFIG["DEFAULT_MODEL_NAME"], help="Path to the trained model file.")

    args = parser.parse_args()
    
    # --- PyTorch Dependency Check ---
    import importlib.util
    REQUIRED_LIBS = {
        'pipeline': ['torch', 'torchvision', 'optuna', 'sklearn'], 
        'tune': ['torch', 'torchvision', 'optuna'],
        'train': ['torch', 'torchvision'], 
        'evaluate': ['torch', 'torchvision', 'sklearn'], 
        'predict': ['torch', 'torchvision']
    }
    missing_libs = [lib for lib in REQUIRED_LIBS.get(args.mode, []) if importlib.util.find_spec(lib) is None]
    if missing_libs:
        print(f"Error: Missing required libraries for '{args.mode}' mode: {', '.join(missing_libs)}", file=sys.stderr)
        print("Please install them. For example:", file=sys.stderr)
        if any(lib in missing_libs for lib in ['torch', 'torchvision']): print("  pip install torch torchvision", file=sys.stderr)
        if 'optuna' in missing_libs: print("  pip install optuna", file=sys.stderr)
        if 'sklearn' in missing_libs: print("  pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    if hasattr(args, 'img_width') and hasattr(args, 'img_height'):
        if args.img_width < args.img_height:
            parser.error("Image width must be greater than or equal to image height.")

    setup_logging(args.verbose)
    
    run_action(args)

def run_action(args: argparse.Namespace):
    """Calls the appropriate function based on the chosen mode."""
    if args.mode == 'pipeline': run_pipeline(args)
    elif args.mode == 'tune': tune_model(args)
    elif args.mode == 'train': train_model(args)
    elif args.mode == 'evaluate': evaluate_model(args)
    elif args.mode == 'predict': predict_image(args)

if __name__ == '__main__':
    main()
