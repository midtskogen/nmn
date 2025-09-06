#!/usr/bin/env python3
"""
A compact script for meteor image classification using a pre-trained EfficientNet model.
This script is a lightweight version of the prediction functionality in classify.py.
"""
import os
import sys
import argparse
import pathlib
import io
from typing import Optional, List

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    import zstandard as zstd
except ImportError as e:
    print(f"Error: A required library is missing. ({e})", file=sys.stderr)
    print("Please install the necessary packages by running:", file=sys.stderr)
    print("pip install torch torchvision Pillow zstandard", file=sys.stderr)
    sys.exit(1)

def resolve_model_path(model_filenames: List[str]) -> Optional[str]:
    """
    Finds the first available model file from a list of candidates.
    Searches in the current directory and the user's nmn/model directory.
    """
    for model_filename in model_filenames:
        # 1. Check current directory
        if os.path.exists(model_filename):
            return model_filename
        # 2. Check user's nmn/model directory
        user_model_path = pathlib.Path.home() / 'nmn' / 'model' / model_filename
        if os.path.exists(user_model_path):
            return str(user_model_path)
    return None

def load_prediction_model(model_path: str) -> nn.Module:
    """
    Initializes and loads the specified EfficientNet model for prediction.
    Handles both compressed (.zst) and uncompressed (.pth) model files.
    """
    # Initialize the model structure
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    
    buffer = None
    try:
        if model_path.endswith('.zst'):
            # Decompress the model weights from the .zst file into a memory buffer
            print(f"Decompressing model: {os.path.basename(model_path)}", file=sys.stderr)
            with open(model_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    buffer = io.BytesIO(reader.read())
        else:
            # For uncompressed .pth files, the path itself can be used by torch.load
            buffer = model_path
    except Exception as e:
        print(f"Error: Failed to read or decompress model file '{model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Load the state dictionary from the buffer or path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(buffer, map_location=device))
    model.eval()
    return model

def main():
    """Parses arguments, loads the model, and predicts on input images."""
    # --- Configuration ---
    MODEL_FILENAMES_TO_TRY = [
        'meteor_efficientnet_b0_model_clustered.pth.zst', # Preferred (compressed, clustered)
        'meteor_efficientnet_b0_model.pth'               # Fallback (uncompressed, unclustered)
    ]
    IMG_HEIGHT = 96
    IMG_WIDTH = 192

    parser = argparse.ArgumentParser(
        description="Classify meteor images using a pre-trained EfficientNet model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'files',
        nargs='+',
        help="Path to one or more image or video files to classify."
    )
    args = parser.parse_args()

    # Find and load the model
    model_path = resolve_model_path(MODEL_FILENAMES_TO_TRY)
    if not model_path:
        print("Error: Could not find a suitable model file.", file=sys.stderr)
        print("Please place one of the following in the current directory or in '~/nmn/model/':", file=sys.stderr)
        for name in MODEL_FILENAMES_TO_TRY:
            print(f"  - {name}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading model from: {model_path}", file=sys.stderr)
    model = load_prediction_model(model_path)
    device = next(model.parameters()).device

    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process each file
    with torch.no_grad():
        for file_path in args.files:
            path_to_process = file_path
            try:
                # If a video file is given, automatically switch to the corresponding .jpg
                if any(file_path.lower().endswith(ext) for ext in ['.webm', '.mp4', '.avi', '.mov', '.mkv']):
                    base, _ = os.path.splitext(file_path)
                    path_to_process = base + '.jpg'
                    print(f"Info: Video file detected, processing '{os.path.basename(path_to_process)}' instead.", file=sys.stderr)

                img = Image.open(path_to_process).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                
                # The model is trained to output P(non_meteor), so P(meteor) is 1.0 - output
                pred_prob_non_meteor = model(tensor).item()
                final_prob_meteor = 1.0 - pred_prob_non_meteor
                
                # Use the basename of the originally provided file for the output
                base_name = os.path.basename(os.path.splitext(file_path)[0])
                print(f"{base_name}: {final_prob_meteor:.6f}")

            except FileNotFoundError:
                print(f"Error: File not found at '{path_to_process}'", file=sys.stderr)
            except Exception as e:
                print(f"Error: Could not process file '{file_path}': {e}", file=sys.stderr)

if __name__ == '__main__':
    main()

