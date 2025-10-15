#!/usr/bin/env python3
"""
Main script for Dhan-Shomadhan Rice Disease Dataset
- Fix spelling errors in CSV
- Convert dataset to YOLO format
- Train YOLOv8 model
"""

import os
import pandas as pd
import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse


def fix_spelling_errors(csv_path, output_path):
    """Fix spelling errors in the CSV file."""
    print("Fixing spelling errors in CSV...")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Define spelling corrections
    corrections = {
        'Browon Spot(Feild Background)': 'Brown Spot(Field Background)',
        'Browon Spot(white Background)': 'Brown Spot(White Background)',
        'Leaf Scaled(Feild Background)': 'Leaf Scald(Field Background)',
        'Leaf Scaled(white Background)': 'Leaf Scald(White Background)',
        'Rice Blast(Feild Background)': 'Rice Blast(Field Background)',
        'Rice Blast(white Background)': 'Rice Blast(White Background)',
        'Rice Turgro(Feild Background)': 'Rice Tungro(Field Background)',
        'Rice Tungro(white Background)': 'Rice Tungro(White Background)',
        'Sheath Blight(Feild Background)': 'Sheath Blight(Field Background)',
        'Shath Blight(white Background)': 'Sheath Blight(White Background)'
    }

    # Apply corrections
    df['Diseases'] = df['Diseases'].replace(corrections)

    # Save corrected CSV
    df.to_csv(output_path, index=False)
    print(f"Corrected CSV saved to: {output_path}")

    return df


def get_image_path(image_name, base_dir):
    """Find the actual image path in the directory structure."""
    # Determine background type and disease from filename
    if image_name.startswith('bsf_'):
        return os.path.join(base_dir, 'Field Background', 'Browon Spot', image_name)
    elif image_name.startswith('bs_wb_'):
        return os.path.join(base_dir, 'White Background', 'Brown Spot', image_name)
    elif image_name.startswith('lsf_'):
        return os.path.join(base_dir, 'Field Background', 'Leaf Scaled', image_name)
    elif image_name.startswith('ls_wb_'):
        return os.path.join(base_dir, 'White Background', 'Leaf Scaled', image_name)
    elif image_name.startswith('rbf_'):
        return os.path.join(base_dir, 'Field Background', 'Rice Blast', image_name)
    elif image_name.startswith('rb_wb_'):
        return os.path.join(base_dir, 'White Background', 'Rice Blast', image_name)
    elif image_name.startswith('rtf_'):
        return os.path.join(base_dir, 'Field Background', 'Rice Turgro', image_name)
    elif image_name.startswith('rt_wb_'):
        return os.path.join(base_dir, 'White Background', 'Rice Tungro', image_name)
    elif image_name.startswith('sbf_'):
        return os.path.join(base_dir, 'Field Background', 'Sheath Blight', image_name)
    elif image_name.startswith('sb_wb_'):
        return os.path.join(base_dir, 'White Background', 'Shath Blight', image_name)
    else:
        return None


def extract_disease_name(full_disease):
    """Extract clean disease name from full disease string."""
    # Remove background info in parentheses
    disease = full_disease.split('(')[0].strip()
    return disease


def convert_to_yolo_format(df, base_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Convert dataset to YOLO format with train/val/test splits."""
    print("Converting to YOLO format...")

    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Extract unique disease classes
    df['disease_clean'] = df['Diseases'].apply(extract_disease_name)
    class_names = sorted(df['disease_clean'].unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"Found {len(class_names)} disease classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42, stratify=df['disease_clean'])
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio/(val_ratio + test_ratio), random_state=42, stratify=temp_df['disease_clean'])

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")

    # Process each split
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} split...")
        processed = 0
        skipped = 0

        for idx, row in split_df.iterrows():
            image_name = row['pictureName']
            disease_full = row['Diseases']
            disease_clean = row['disease_clean']
            class_idx = class_to_idx[disease_clean]

            # Get source image path
            src_image_path = get_image_path(image_name, base_dir)

            if not src_image_path or not os.path.exists(src_image_path):
                skipped += 1
                continue

            # Copy image
            dst_image_path = output_path / split_name / 'images' / image_name
            shutil.copy2(src_image_path, dst_image_path)

            # Create label file (classification format: just class index)
            # For object detection, we would need bounding boxes
            # For now, creating a simple whole-image label
            label_name = image_name.replace('.jpg', '.txt')
            label_path = output_path / split_name / 'labels' / label_name

            # Get image dimensions
            try:
                with Image.open(src_image_path) as img:
                    width, height = img.size

                # Create YOLO format label: class x_center y_center width height (normalized)
                # For classification, we consider the whole image as one object
                x_center = 0.5
                y_center = 0.5
                bbox_width = 1.0
                bbox_height = 1.0

                with open(label_path, 'w') as f:
                    f.write(f"{class_idx} {x_center} {y_center} {bbox_width} {bbox_height}\n")

                processed += 1
            except Exception as e:
                print(f"  Error processing {image_name}: {e}")
                skipped += 1
                continue

        print(f"  Processed: {processed}, Skipped: {skipped}")

    return class_names, class_to_idx


def create_data_yaml(output_dir, class_names):
    """Create YOLO data configuration file."""
    print("\nCreating data.yaml configuration...")

    # Get absolute path
    dataset_path = os.path.abspath(output_dir)

    data_config = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to: {yaml_path}")
    print(f"Dataset path: {dataset_path}")

    return yaml_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Prepare Dhan-Shomadhan dataset and train YOLOv8')
    parser.add_argument('--csv', type=str, default='Dhan-Shomadhan_picture_Information.csv',
                        help='Input CSV file path')
    parser.add_argument('--base-dir', type=str, default='Dhan-Shomadhan',
                        help='Base directory containing images')
    parser.add_argument('--output-dir', type=str, default='yolo_dataset',
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train', 'eval', 'inference', 'visualize'],
                        help='Execution mode: all (run everything), train (training only), eval (evaluation only), inference (inference only), visualize (visualization only)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--model-path', type=str, default='outputs/models/rice_disease_detection/weights/best.pt',
                        help='Path to trained model for evaluation/inference')
    parser.add_argument('--inference-dir', type=str, default='yolo_dataset/test/images',
                        help='Directory containing images for inference')
    args = parser.parse_args()

    print("=" * 70)
    print("Dhan-Shomadhan Rice Disease Dataset - Main Pipeline")
    print(f"Mode: {args.mode}")
    print("=" * 70)

    yaml_path = 'data.yaml'

    # Step 1: Data Preparation (always run unless mode is eval/inference/visualize only)
    if args.mode in ['all', 'train']:
        print("\n" + "=" * 70)
        print("STEP 1: Data Preparation")
        print("=" * 70)

        # Fix spelling errors
        corrected_csv = 'Dhan-Shomadhan_picture_Information_corrected.csv'
        df = fix_spelling_errors(args.csv, corrected_csv)

        # Convert to YOLO format
        class_names, _ = convert_to_yolo_format(
            df, args.base_dir, args.output_dir
        )

        # Create data.yaml
        yaml_path = create_data_yaml(args.output_dir, class_names)

        print("\nData preparation complete!")
        print(f"Corrected CSV: {corrected_csv}")
        print(f"YOLO dataset: {args.output_dir}")
        print(f"Configuration: {yaml_path}")

    # Step 2: Train model
    if args.mode in ['all', 'train']:
        print("\n" + "=" * 70)
        print("STEP 2: Model Training")
        print("=" * 70)

        try:
            from ultralytics import YOLO

            # Initialize model
            model = YOLO('yolov11n.pt')

            # Train
            results = model.train(
                data=yaml_path,
                epochs=args.epochs,
                imgsz=640,
                batch=16,
                device='cuda' if os.path.exists('/proc/driver/nvidia/version') else 'cpu',
                project='outputs/models',
                optimizer='AdamW',
                lr0=0.001,  
                name='rice_disease_detection',
                save=True,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                degrees=15.0, flipud=0.5, fliplr=0.5,
                mosaic=1.0, mixup=0.15,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                plots=True
            )

            print("\nTraining complete!")
            print(f"Model saved to: outputs/models/rice_disease_detection")
            args.model_path = 'outputs/models/rice_disease_detection/weights/best.pt'

        except ImportError:
            print("\nUltralytics YOLO not installed. Please install with:")
            print("  pip install ultralytics")
            return
        except Exception as e:
            print(f"\nError during training: {e}")
            return

    # Step 3: Model Evaluation
    if args.mode in ['all', 'eval']:
        print("\n" + "=" * 70)
        print("STEP 3: Model Evaluation")
        print("=" * 70)

        try:
            from model_evaluation import ModelEvaluator

            if not os.path.exists(args.model_path):
                print(f"Model file not found: {args.model_path}")
                print("Please train a model first or specify correct --model-path")
                return

            evaluator = ModelEvaluator(args.model_path, yaml_path)
            evaluator.run_complete_evaluation()

            print("\nEvaluation complete!")

        except ImportError as e:
            print(f"\nError importing model_evaluation: {e}")
        except Exception as e:
            print(f"\nError during evaluation: {e}")

    # Step 4: Inference
    if args.mode in ['all', 'inference']:
        print("\n" + "=" * 70)
        print("STEP 4: Inference")
        print("=" * 70)

        try:
            from inference import RiceDiseaseDetector

            if not os.path.exists(args.model_path):
                print(f"Model file not found: {args.model_path}")
                print("Please train a model first or specify correct --model-path")
                return

            if not os.path.exists(args.inference_dir):
                print(f"Inference directory not found: {args.inference_dir}")
                print("Please specify correct --inference-dir")
                return

            detector = RiceDiseaseDetector(args.model_path, confidence_threshold=0.5)
            detector.detect_batch(args.inference_dir, output_dir='outputs/inference_results')

            print("\nInference complete!")
            print("Results saved to: outputs/inference_results")

        except ImportError as e:
            print(f"\nError importing inference: {e}")
        except Exception as e:
            print(f"\nError during inference: {e}")

    # Step 5: Visualization
    if args.mode in ['all', 'visualize']:
        print("\n" + "=" * 70)
        print("STEP 5: Results Visualization")
        print("=" * 70)

        try:
            from visualize_results import ResultsVisualizer

            if not os.path.exists(args.model_path):
                print(f"Model file not found: {args.model_path}")
                print("Please train a model first or specify correct --model-path")
                return

            visualizer = ResultsVisualizer(results_dir='outputs/models/rice_disease_detection')
            visualizer.generate_report(model_path=args.model_path)

            print("\nVisualization complete!")

        except ImportError as e:
            print(f"\nError importing visualize_results: {e}")
        except Exception as e:
            print(f"\nError during visualization: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print("\nUsage examples:")
    print("  # Run everything:")
    print("  python main.py --mode all --epochs 50")
    print("\n  # Train only:")
    print("  python main.py --mode train --epochs 50")
    print("\n  # Evaluate only:")
    print("  python main.py --mode eval --model-path path/to/best.pt")
    print("\n  # Inference only:")
    print("  python main.py --mode inference --model-path path/to/best.pt --inference-dir path/to/images")
    print("\n  # Visualize only:")
    print("  python main.py --mode visualize --model-path path/to/best.pt")


if __name__ == "__main__":
    main()
