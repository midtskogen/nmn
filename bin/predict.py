#!/usr/bin/env python3
"""
A compact script for meteor image classification using a pre-trained EfficientNet model.
"""
import os
import sys
import argparse
import pathlib
import io
from typing import Optional, List
import contextlib
import tempfile

@contextlib.contextmanager
def suppress_nnpack_warnings():
    """
    A context manager to robustly suppress harmless NNPACK warnings by redirecting
    the stderr file descriptor at the OS level.
    """
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_f:
            os.dup2(temp_f.fileno(), stderr_fd)
            yield
            temp_f.seek(0)
            captured_output = temp_f.read()
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        
    is_harmless = "NNPACK" in captured_output and "fatal" not in captured_output.lower()
    if not is_harmless and captured_output:
        sys.stderr.write(captured_output)

try:
    with suppress_nnpack_warnings():
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from PIL import Image
        import zstandard as zstd
except ImportError as e:
    sys.exit(
        f"Error: A required library is missing ({e}).\n"
        "Please run: pip install torch torchvision Pillow zstandard"
    )

def resolve_model_path(filenames: List[str]) -> Optional[str]:
    """Finds the first available model file from a list of candidates."""
    for filename in filenames:
        if os.path.exists(filename): return filename
        user_path = pathlib.Path.home() / 'nmn' / 'model' / filename
        if user_path.exists(): return str(user_path)
    return None

def load_model(model_path: str):
    """Loads a model, handling compressed and standard PyTorch formats."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {os.path.basename(model_path)} (device: {device})", file=sys.stderr)

    model = models.efficientnet_b0()
    model.classifier = nn.Sequential(nn.Linear(model.classifier[1].in_features, 1), nn.Sigmoid())
    
    buffer = model_path
    if model_path.endswith('.zst'):
        try:
            # Use a more robust streaming decompressor
            with open(model_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    buffer = io.BytesIO(reader.read())
        except Exception as e:
            sys.exit(f"Error: Failed to decompress model '{model_path}': {e}")
    
    model.load_state_dict(torch.load(buffer, map_location=device))
    
    return model.eval().to(device)

def main():
    """Parses arguments, loads the model, and predicts on input images."""
    MODEL_FILES = ['meteor_efficientnet_b0_model_clustered.pth.zst', 'meteor_efficientnet_b0_model.pth']
    IMG_SIZE = (96, 192)
    VIDEO_EXTS = ['.webm', '.mp4', '.avi', '.mov', '.mkv']

    parser = argparse.ArgumentParser(description="Classify meteor images.")
    parser.add_argument('files', nargs='+', help="Path to one or more image/video files.")
    args = parser.parse_args()

    model_path = resolve_model_path(MODEL_FILES)
    if not model_path:
        sys.exit(
            "Error: Could not find a model file.\nPlease place one of the following in the current directory or '~/nmn/model/':\n"
            + "\n".join([f"  - {name}" for name in MODEL_FILES])
        )
    
    model = load_model(model_path)
    device = next(model.parameters()).device
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad(), suppress_nnpack_warnings():
        for file_path in args.files:
            try:
                base, ext = os.path.splitext(file_path)
                img_path = (base + '.jpg') if ext.lower() in VIDEO_EXTS else file_path
                
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                
                pred_prob = model(tensor).item()
                print(f"{os.path.basename(base)}: {1.0 - pred_prob:.6f}")

            except FileNotFoundError:
                print(f"Error: Input file not found at '{img_path}'", file=sys.stderr)
            except Exception as e:
                print(f"Error: Could not process '{file_path}': {e}", file=sys.stderr)

if __name__ == '__main__':
    main()

