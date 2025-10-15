#!/usr/bin/env python3
"""
YOLOv8 Rice Disease Detection - Orchestrator
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


def check_dataset():
    """Check dataset layout and required files."""
    print("Checking dataset...")
    
    yolo_dataset = Path("outputs/yolo_dataset")
    if not yolo_dataset.exists():
        print(f"Dataset directory not found: {yolo_dataset}")
        return False
    
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']
    for dir_path in required_dirs:
        full_path = yolo_dataset / dir_path
        if not full_path.exists():
            print(f"Missing dataset subdirectory: {full_path}")
            return False
    
    # Check data file
    data_yaml = Path("data.yaml")
    if not data_yaml.exists():
        print(f"Config file not found: {data_yaml}")
        return False
    
    print("Dataset check passed")
    return True

def train_model(args):
    """Train model."""
    print("Starting training...")

    
    if not check_dataset():
        return False
    
    try:
        # Run training script using current interpreter
        result = subprocess.run(
            [sys.executable, "train_yolov8.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode == 0:
            print("Training completed.")
            print(result.stdout)
            return True
        else:
            print("Training failed.")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

def evaluate_model(args):
    """Evaluate model."""
    print("Evaluating model...")
    
    model_path = "outputs/models/rice_disease_detection6/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Train a model first.")
        return False
    
    try:
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(model_path)
        
        # Evaluate
        metrics = model.val(data="data.yaml")
        
        print("Metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return False

def run_inference(args):
    """Run inference."""
    print("Running inference...")
    
    model_path = "outputs/models/rice_disease_detection8/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Train a model first.")
        return False
    
    if not args.input:
        print("Please provide an input image or directory.")
        return False
    
    try:
        from inference import RiceDiseaseDetector
        
        # Create detector
        detector = RiceDiseaseDetector(model_path, confidence_threshold=args.confidence)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single image
            print(f"Image: {input_path}")
            detections, annotated_image = detector.detect_image(input_path)
            
            print(f"Detections: {len(detections)}")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class_name']} (confidence: {det['confidence']:.3f})")
                
        elif input_path.is_dir():
            # Batch
            print(f"Directory: {input_path}")
            detector.detect_batch(input_path)
            
        else:
            print(f"Input path not found: {input_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return False

def visualize_results(args):
    """Visualize results."""
    print("Visualizing results...")
    
    model_path = "outputs/models/rice_disease_detection6/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Train a model first.")
        return False
    
    try:
        # Run visualization script
        result = subprocess.run(
            [sys.executable, "visualize_results.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        
        if result.returncode == 0:
            print("Visualization completed.")
            print(result.stdout)
            return True
        else:
            print("Visualization failed.")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        return False

def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="YOLOv8 Rice Disease Detection")
    parser.add_argument("command", choices=["train", "eval", "infer", "visualize", "all"],
                       help="Operation to run")
    parser.add_argument("--input", "-i", help="Input image or directory for inference")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    print("YOLOv8 Rice Disease Detection")
    print("=" * 50)
    
    success = True
    
    if args.command == "train":
        success = train_model(args)
        
    elif args.command == "eval":
        success = evaluate_model(args)
        
    elif args.command == "infer":
        success = run_inference(args)
        
    elif args.command == "visualize":
        success = visualize_results(args)
        
    elif args.command == "all":
        print("Running full pipeline...")
        
        # 1. Train
        print("\n1) Train")
        if not train_model(args):
            success = False
            print("Training failed. Aborting.")
            return
        
        # 2. Evaluate
        print("\n2) Evaluate")
        if not evaluate_model(args):
            success = False
        
        # 3. Visualize
        print("\n3) Visualize")
        if not visualize_results(args):
            success = False
        
        # 4. Example inference
        print("\n4) Example inference")
        test_dir = "outputs/yolo_dataset/test/images"
        if os.path.exists(test_dir):
            args.input = test_dir
            if not run_inference(args):
                success = False
        else:
            print(f"Test image directory not found: {test_dir}")
    
    if success:
        print("\nAll operations completed.")
        print("Artifacts:")
        print("  - Weights: outputs/models/rice_disease_detection/weights/")
        print("  - Training: outputs/models/rice_disease_detection/")
        print("  - Results: outputs/results/")
    else:
        print("\nSome operations failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
