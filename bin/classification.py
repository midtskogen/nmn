#!/usr/bin/env python3
"""
A professional-grade script for meteor image classification with five modes:
1. pipeline: Runs the full tune -> train -> evaluate pipeline across multiple resolutions.
2. tune:     Uses KerasTuner to find the best model architecture for a single resolution.
3. train:    Trains a model using either default or tuned hyperparameters.
4. evaluate: Analyzes a trained model to find the optimal classification threshold.
5. predict:  Uses a trained model to classify one or more images.

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
from typing import Dict, Any, List
import numpy as np
import random
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed, so the script doesn't crash.
    def tqdm(iterable, *args, **kwargs):
        logging.info("tqdm not found. To see progress bars, please run 'pip install tqdm'")
        return iterable

# --- Default Configuration ---
CONFIG = {
    "DEFAULT_IMG_HEIGHT": 128,
    "DEFAULT_IMG_WIDTH": 128,
    "DEFAULT_MODEL_NAME": 'meteor_model.keras',
    "PARAMS_FILE": 'best_params.json'
}

# --- Utility Functions ---

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

    # --- Optional, On-the-fly Dataset Balancing ---
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

# --- Core Logic Functions (Refactored for Pipeline) ---

def tune_model(args: argparse.Namespace):
    """Public-facing wrapper for the tuning logic."""
    params_path = os.path.join(os.getcwd(), CONFIG["PARAMS_FILE"])
    _run_tuning(args.positive_dir, args.negative_dir, (args.img_height, args.img_width),
                args.batch_size, args.epochs, os.getcwd(), params_path, args.balance)
    
    # MODIFIED: Add helpful next-step message
    script_name = os.path.basename(sys.argv[0])
    print("\n" + "="*20 + " Next Step " + "="*20)
    print(f"Tuning complete. Best parameters saved to '{params_path}'")
    print("\nTo train a model with these optimal parameters, run:")
    print(f"  python {script_name} train {args.positive_dir} {args.negative_dir} --params-file {params_path} "
          f"--img-width {args.img_width} --img-height {args.img_height}")

def _run_tuning(positive_dir, negative_dir, image_size, batch_size, epochs, output_dir, params_path, balance_dataset):
    import tensorflow as tf
    import keras_tuner as kt
    from tensorflow.keras import layers, models
    from tensorflow.keras.utils import image_dataset_from_directory
    
    img_height, img_width = image_size
    logging.info(f"--- Starting Hyperparameter Tuning for resolution {img_width}x{img_height} ---")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_directories(positive_dir, negative_dir, temp_dir, balance_dataset=balance_dataset)
        train_dataset = image_dataset_from_directory(
            data_dir, validation_split=0.2, subset="training", seed=123, image_size=image_size,
            batch_size=batch_size, label_mode='binary')
        validation_dataset = image_dataset_from_directory(
            data_dir, validation_split=0.2, subset="validation", seed=123, image_size=image_size,
            batch_size=batch_size, label_mode='binary')
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        def build_model(hp: kt.HyperParameters) -> models.Model:
            model = models.Sequential([
                layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2)])
            for i in range(hp.Int('num_conv_layers', 2, 4)):
                model.add(layers.Conv2D(filters=hp.Int(f'conv_{i}_filters', min_value=32, max_value=128, step=32),
                                       kernel_size=(3, 3), padding='same', activation='relu'))
                model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(units=hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'))
            model.add(layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
            model.add(layers.Dense(1, activation='sigmoid'))
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                          loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
            return model

        tuner = kt.Hyperband(
            build_model, objective='val_accuracy', max_epochs=epochs, factor=3,
            directory=os.path.join(output_dir, 'tuning_dir'),
            project_name=f'meteor_{img_width}x{img_height}', overwrite=True)
        tuner.search(train_dataset, epochs=epochs, validation_data=validation_dataset,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logging.info(f"\n--- Tuning Complete for {img_width}x{img_height} ---\n" +
                     f"Optimal number of conv layers: {best_hps.get('num_conv_layers')}\n" +
                     f"Optimal learning rate: {best_hps.get('learning_rate')}\n")
        
        params_to_save = {key: val.item() if isinstance(val, np.generic) else val for key, val in best_hps.values.items()}
        with open(params_path, 'w') as f: json.dump(params_to_save, f, indent=4)
        logging.info(f"Best parameters saved to {params_path}")
    return params_path

def train_model(args: argparse.Namespace):
    """Public-facing wrapper for the training logic."""
    model_path = os.path.join(os.getcwd(), args.output)
    _run_training(args.positive_dir, args.negative_dir, (args.img_height, args.img_width),
                  args.batch_size, args.epochs, args.params_file, model_path, args.balance)

def _run_training(positive_dir, negative_dir, image_size, batch_size, epochs, params_file, model_path, balance_dataset):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.utils import image_dataset_from_directory
    
    img_height, img_width = image_size
    logging.info(f"--- Starting Model Training for resolution {img_width}x{img_height} ---")
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
        train_dataset = image_dataset_from_directory(
            data_dir, validation_split=0.2, subset="training", seed=123, image_size=image_size,
            batch_size=batch_size, label_mode='binary')
        validation_dataset = image_dataset_from_directory(
            data_dir, validation_split=0.2, subset="validation", seed=123, image_size=image_size,
            batch_size=batch_size, label_mode='binary')
        class_names = train_dataset.class_names
        logging.info(f"Classes found: {class_names}")
        meteor_class_index = class_names.index('meteor')
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        model = models.Sequential()
        model.add(layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
        model.add(layers.RandomFlip("horizontal_and_vertical"))
        model.add(layers.RandomRotation(0.2))
        for i in range(params.get('num_conv_layers', 3)):
            model.add(layers.Conv2D(params.get(f'conv_{i}_filters', 64), (3, 3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(params.get('dense_units', 128), activation='relu'))
        model.add(layers.Dropout(params.get('dropout_rate', 0.5)))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 1e-3)),
                      loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        model.summary(print_fn=logging.info)
        logging.info("Starting training with automated stopping...")
        callbacks = [EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True),
                     ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)]
        model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks)
    logging.info(f"Training complete. Best model saved to '{model_path}'")
    with open('class_indices.txt', 'w') as f: f.write(str(meteor_class_index))
    return model_path

def evaluate_model(args: argparse.Namespace):
    """Public-facing wrapper for the evaluation logic."""
    _run_evaluation(args.positive_dir, args.negative_dir, (args.img_height, args.img_width),
                    args.batch_size, args.model_file)

def _run_evaluation(positive_dir, negative_dir, image_size, batch_size, model_path):
    import tensorflow as tf
    from sklearn.metrics import precision_recall_curve, confusion_matrix
    from tensorflow.keras.utils import image_dataset_from_directory
    
    img_height, img_width = image_size
    logging.info(f"--- Starting Model Evaluation for {img_width}x{img_height} ---")
    if not os.path.isfile(model_path):
        logging.error(f"Model file not found at '{model_path}'"); return None

    model = tf.keras.models.load_model(model_path)
    results = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = prepare_data_directories(positive_dir, negative_dir, temp_dir, balance_dataset=False)
        validation_dataset = image_dataset_from_directory(
            data_dir, validation_split=0.2, subset="validation", seed=123, image_size=image_size,
            batch_size=batch_size, label_mode='binary', shuffle=False)
        logging.info("Extracting true labels and predicting probabilities...")
        y_true = np.concatenate([y for x, y in validation_dataset], axis=0).flatten()
        y_pred_probs = model.predict(validation_dataset).flatten()
        try:
            with open('class_indices.txt', 'r') as f: meteor_class_index = int(f.read())
        except FileNotFoundError:
            logging.warning("class_indices.txt not found."); meteor_class_index = -1
        if meteor_class_index == 0:
            logging.info("Keras labeled 'meteor' as class 0. Inverting probabilities and labels for evaluation.")
            y_pred_probs = 1 - y_pred_probs
            y_true = 1 - y_true
        logging.info("Calculating metrics across all thresholds...")
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        best_f1, best_precision, best_recall = f1_scores[best_f1_idx], precision[best_f1_idx], recall[best_f1_idx]
        y_pred_binary = (y_pred_probs >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        results = {
            "resolution": f"{img_width}x{img_height}", "f1_score": float(best_f1),
            "precision": float(best_precision), "recall": float(best_recall), "threshold": float(best_threshold)}
        logging.info(f"\n--- Evaluation Report for {img_width}x{img_height} ---")
        print(f"Optimal Threshold: {results['threshold']:.4f}")
        print(f"  - F1-Score:      {results['f1_score']:.4f} (Best balance between precision and recall)")
        print(f"  - Precision:     {results['precision']:.4f} (Of all predicted meteors, {results['precision']:.2%} were real)")
        print(f"  - Recall:        {results['recall']:.4f} (Of all real meteors, {results['recall']:.2%} were found)")
    return results

def predict_image(args: argparse.Namespace):
    """Classifies one or more images using a trained model."""
    import tensorflow as tf
    if not os.path.isfile(args.model_file):
        logging.error(f"Model file not found at '{args.model_file}'"); sys.exit(1)
    logging.info(f"Loading model: {args.model_file}")
    model = tf.keras.models.load_model(args.model_file)
    try:
        with open('class_indices.txt', 'r') as f: meteor_class_index = int(f.read())
    except FileNotFoundError:
        logging.error("Error: class_indices.txt not found."); sys.exit(1)
    model_input_shape = model.input_shape
    img_height, img_width = model_input_shape[1], model_input_shape[2]
    logging.info(f"Model expects input resolution: {img_width}x{img_height}")
    for image_path in args.image_files:
        if not os.path.isfile(image_path):
            logging.warning(f"Skipping '{image_path}': file not found."); continue
        img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array, verbose=0)
        score = predictions[0][0]
        meteor_probability = 1 - score if meteor_class_index == 0 else score
        print(f"{os.path.basename(image_path)}: {meteor_probability:.6f}")

def run_pipeline(args: argparse.Namespace):
    """Runs the full tune->train->evaluate pipeline across multiple resolutions."""
    heights = [64, 96, 128]
    widths = [96, 128, 160, 192, 224, 256]
    resolutions_to_test = sorted([(w, h) for w in widths for h in heights if w >= h])
    
    pipeline_dir = "pipeline_results"
    os.makedirs(pipeline_dir, exist_ok=True)
    summary_file_path = os.path.join(pipeline_dir, "summary.json")
    
    all_results = []
    completed_resolutions = set()

    if args.resume:
        logging.info(f"Attempting to resume pipeline from '{pipeline_dir}'...")
        if os.path.exists(summary_file_path):
            try:
                with open(summary_file_path, 'r') as f:
                    all_results = json.load(f)
                for result in all_results:
                    completed_resolutions.add(result['resolution'])
                if completed_resolutions:
                    logging.info(f"Resumed {len(completed_resolutions)} completed resolutions from summary.json: {', '.join(sorted(completed_resolutions))}")
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
                                                   args.batch_size, model_path)
                    if eval_results:
                        all_results.append(eval_results)
                        completed_resolutions.add(res_str)
                        logging.info(f"Successfully recovered results for {res_str}.")
                except Exception as e:
                    logging.error(f"Failed to recover results for {res_str} due to an error during evaluation: {e}")
            
        if all_results:
            with open(summary_file_path, 'w') as f:
                json.dump(all_results, f, indent=4)
    else:
        if os.path.exists(pipeline_dir):
            logging.info(f"Starting new pipeline. Removing existing results in '{pipeline_dir}'...")
            shutil.rmtree(pipeline_dir)
        os.makedirs(pipeline_dir, exist_ok=True)
        logging.info(f"Starting new pipeline. Results will be stored in '{pipeline_dir}'")
    
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
                                           args.batch_size, model_path)
            if eval_results:
                all_results.append(eval_results)
                with open(summary_file_path, 'w') as f:
                    json.dump(all_results, f, indent=4)
                logging.info(f"Successfully completed and saved results for {res_str}")

        except Exception as e:
            logging.error(f"Pipeline failed for resolution {width}x{height}: {e}")
            logging.info("Continuing to next resolution...")
            continue

    if not all_results:
        logging.error("Pipeline finished but no results were generated. Please check logs for errors.")
        return

    all_results.sort(key=lambda x: x['f1_score'], reverse=True)
    best_result = all_results[0]
    
    logging.info("\n" + "="*50 + "\n--- Pipeline Complete: Final Summary ---\n" + "="*50)
    print(f"Best overall performance found for resolution {best_result['resolution']}\n")
    print(f"| {'Resolution':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Optimal Threshold':<20} |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*22}|")
    for result in all_results:
        print(f"| {result['resolution']:<10} | {result['f1_score']:.4f}     | {result['precision']:.2%}     | {result['recall']:.2%}   | {result['threshold']:.4f}               |")
    
    # MODIFIED: Add helpful final message with location of best model and params
    best_res_dir = os.path.join(pipeline_dir, best_result['resolution'])
    print("\n" + "="*20 + " Best Model Location " + "="*20)
    print(f"The best model and its parameters are located in:")
    print(f"  Model:       {os.path.join(best_res_dir, CONFIG['DEFAULT_MODEL_NAME'])}")
    print(f"  Parameters:  {os.path.join(best_res_dir, CONFIG['PARAMS_FILE'])}")


def main():
    parser = argparse.ArgumentParser(description="A tool for meteor classification.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose logging output.")
    
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Operating mode")
    
    # --- Parent Parsers for common arguments ---
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--img-width', type=int, default=CONFIG["DEFAULT_IMG_WIDTH"], help="Image width for processing.")
    parent_parser.add_argument('--img-height', type=int, default=CONFIG["DEFAULT_IMG_HEIGHT"], help="Image height for processing.")
    parent_parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training/evaluation.")
    
    balance_parent_parser = argparse.ArgumentParser(add_help=False)
    balance_parent_parser.add_argument('--balance', action='store_true', help="Enable on-the-fly dataset balancing.")

    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('positive_dir', help="Directory with positive meteor images.")
    dir_parent_parser.add_argument('negative_dir', help="Directory with negative (non-meteor) images.")

    # --- Pipeline Parser ---
    parser_pipeline = subparsers.add_parser('pipeline', help="Run the full tune->train->evaluate workflow.",
                                            parents=[dir_parent_parser, balance_parent_parser],
                                            formatter_class=argparse.RawTextHelpFormatter,
                                            epilog="example:\n  python %(prog)s ./pos/ ./neg/ --balance --resume")
    parser_pipeline.add_argument('--tune-epochs', type=int, default=20, help="Max epochs for each tuning trial.")
    parser_pipeline.add_argument('--train-epochs', type=int, default=100, help="Max epochs for the final training stage.")
    parser_pipeline.add_argument('--batch-size', type=int, default=32, help="Batch size used throughout the pipeline.")
    parser_pipeline.add_argument('--resume', action='store_true', help="Resume a previously stopped pipeline run.")

    # --- Tune Parser ---
    parser_tune = subparsers.add_parser('tune', help="Find the best model hyperparameters.", parents=[parent_parser, balance_parent_parser, dir_parent_parser],
                                        formatter_class=argparse.RawTextHelpFormatter,
                                        epilog="example:\n  python %(prog)s ./pos/ ./neg/ --balance --img-width 192")
    parser_tune.add_argument('--epochs', type=int, default=20, help="Max epochs per tuning trial.")
    
    # --- Train Parser ---
    parser_train = subparsers.add_parser('train', help="Train the model.", parents=[parent_parser, balance_parent_parser, dir_parent_parser],
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         epilog="""examples:\n  python %(prog)s ./pos/ ./neg/ --balance -o v2.keras""")
    parser_train.add_argument('--params-file', default=CONFIG["PARAMS_FILE"], help="JSON file with tuned hyperparameters.")
    parser_train.add_argument('--epochs', type=int, default=100, help="Maximum number of training epochs.")
    parser_train.add_argument('-o', '--output', default=CONFIG["DEFAULT_MODEL_NAME"], help="Output filename for the model.")
    
    # --- Evaluate Parser ---
    parser_evaluate = subparsers.add_parser('evaluate', help="Find the optimal classification threshold.", parents=[parent_parser, dir_parent_parser],
                                            formatter_class=argparse.RawTextHelpFormatter,
                                            epilog="""example:\n  python %(prog)s ./pos/ ./neg/ -m model.keras""")
    parser_evaluate.add_argument('-m', '--model-file', default=CONFIG["DEFAULT_MODEL_NAME"], help="Path to the trained model file.")
    
    # --- Predict Parser ---
    parser_predict = subparsers.add_parser('predict', help="Classify one or more images.", formatter_class=argparse.RawTextHelpFormatter,
                                           epilog="""examples:\n  python %(prog)s image1.jpg\n  python %(prog)s ./folder/*.jpg -m model.keras""")
    parser_predict.add_argument('image_files', nargs='+', help="One or more image file paths to classify.")
    parser_predict.add_argument('-m', '--model-file', default=CONFIG["DEFAULT_MODEL_NAME"], help="Path to the trained model file.")

    args = parser.parse_args()
    
    # --- Dependency Check ---
    import importlib.util
    REQUIRED_LIBS = {
        'pipeline': ['tensorflow', 'keras_tuner', 'sklearn'], 'tune': ['tensorflow', 'keras_tuner'],
        'train': ['tensorflow'], 'evaluate': ['tensorflow', 'sklearn'], 'predict': ['tensorflow']}
    missing_libs = [lib for lib in REQUIRED_LIBS.get(args.mode, []) if importlib.util.find_spec(lib) is None]
    if missing_libs:
        print(f"Error: Missing required libraries for '{args.mode}' mode: {', '.join(missing_libs)}", file=sys.stderr)
        print("Please install them. For example:", file=sys.stderr)
        if 'keras_tuner' in missing_libs: print("  pip install keras-tuner", file=sys.stderr)
        if 'sklearn' in missing_libs: print("  pip install scikit-learn", file=sys.stderr)
        if 'tensorflow' in missing_libs: print("  pip install tensorflow", file=sys.stderr)
        sys.exit(1)

    if hasattr(args, 'img_width') and hasattr(args, 'img_height'):
        if args.img_width < args.img_height:
            parser.error("Image width must be greater than or equal to image height.")

    setup_logging(args.verbose)
    
    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        with suppress_stderr():
            run_action(args)
    else:
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
