#!/usr/bin/env python3
"""
YOLOv8 Rice Disease Detection - Training
"""

import os
import argparse
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def setup_environment():
    """Set up training environment."""
    print("Setting up YOLOv8 training environment...")
    
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return device, output_dir

def validate_dataset(data_yaml_path):
    """Validate dataset configuration."""
    print("Validating dataset configuration...")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Config file not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # Resolve dataset paths
    base_path = Path(data_config['path'])
    train_path = base_path / data_config['train']
    val_path = base_path / data_config['val']
    test_path = base_path / data_config['test']
    
    print(f"Dataset root: {base_path}")
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Test:  {test_path}")
    
    # Count images
    train_images = len(list(train_path.glob('*.jpg')))
    val_images = len(list(val_path.glob('*.jpg')))
    test_images = len(list(test_path.glob('*.jpg')))
    
    print(f"Images - train: {train_images}, val: {val_images}, test: {test_images}")
    print(f"Classes: {data_config['nc']}")
    print(f"Class names: {data_config['names']}")
    
    return data_config

def train_model(data_yaml_path, device, output_dir, epochs=100, patience=20, early_stop_metric: str = "auto", weights: str = 'yolov8n.pt', resume: bool = False, project: str | None = None, name: str | None = None, save_period: int = 10):
    """Train YOLOv8 model."""
    print("Starting YOLOv8 training...")
    
    # Load model/weights
    model = YOLO(weights)
    
    # Training args
    train_args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': 640,
        'batch': 8 if device == 'cpu' else 16,
        'device': device,
        'project': project if project else str(output_dir),
        'name': name if name else 'rice_disease_detection',
        'save': True,
        'save_period': save_period,
        'cache': False,
        'workers': 4 if device == 'cpu' else 8,
        'patience': patience,
        'resume': resume,
        'lr0': 0.003,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.2,
        'crop_fraction': 1.0,
        'val': True,
        'plots': True,
        'verbose': True
    }
    
    print("Training arguments:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Train
    try:
        results = model.train(**train_args)
    except KeyboardInterrupt:
        print("\nInterrupted. Training stopped safely. Use --resume to continue.")
        raise
    
    print("Training finished.")
    return results, model

def evaluate_model(model, data_yaml_path):
    """Evaluate model on validation set."""
    print("Evaluating model...")
    
    # Validation
    metrics = model.val(data=data_yaml_path)
    
    print("Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics

def main():
    """CLI entrypoint."""
    print("YOLOv8 Rice Disease - Training")
    print("=" * 50)
    parser = argparse.ArgumentParser(description="Train YOLOv8 for rice disease with early stopping")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--early_stop_metric", type=str, default="auto", help="Early stop metric (keep 'auto')")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint (with --project/--name)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial weights (e.g., yolov8n.pt or outputs/.../best.pt)")
    parser.add_argument("--project", type=str, default=None, help="Project directory (for resume lookup)")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (for resume lookup)")
    parser.add_argument("--save_period", type=int, default=10, help="Checkpoint save period (epochs)")
    args = parser.parse_args()

    # Environment
    device, output_dir = setup_environment()
    
    # Data config path
    data_yaml_path = "data.yaml"

    try:
        # Validate dataset
        data_config = validate_dataset(data_yaml_path)
        
        # Train
        results, model = train_model(
            data_yaml_path,
            device,
            output_dir,
            epochs=args.epochs,
            patience=args.patience,
            early_stop_metric=args.early_stop_metric,
            weights=args.weights,
            resume=args.resume,
            project=args.project,
            name=args.name,
            save_period=args.save_period,
        )
        
        # Evaluate
        metrics = evaluate_model(model, data_yaml_path)
        
        # Report best model
        best_model_path = output_dir / "rice_disease_detection" / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"Best model saved to: {best_model_path}")
        
        print("\nTraining complete.")
        print("Artifacts:")
        print(f"  - Runs:   {output_dir / 'rice_disease_detection'}")
        print(f"  - Weights:{output_dir / 'rice_disease_detection' / 'weights'}")
        print(f"  - Plots:  {output_dir / 'rice_disease_detection' / 'results.png'}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
