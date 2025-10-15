#!/usr/bin/env python3
"""
YOLOv8 Rice Disease Detection - Demo
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def demo_without_model():
    """Text-based demo without a trained model."""
    print("YOLOv8 Rice Disease Detection - Demo")
    print("=" * 50)
    
    print("\nFeatures:")
    print("1. Detect five rice leaf diseases:")
    print("   - Brown Spot (褐斑病)")
    print("   - Leaf Scald (叶鞘腐败病)")  
    print("   - Rice Blast (稻瘟病)")
    print("   - Rice Tungro (东格鲁病毒病)")
    print("   - Sheath Blight (纹枯病)")
    
    print("\nUsage:")
    print("1. Train:")
    print("   python main.py train")
    print("   或")
    print("   python train_yolov8.py")
    
    print("\n2. Evaluate:")
    print("   python main.py eval")
    
    print("\n3. Inference:")
    print("   python main.py infer --input image.jpg")
    print("   python main.py infer --input image_directory/")
    
    print("\n4. Visualize:")
    print("   python main.py visualize")
    
    print("\n5. Full pipeline:")
    print("   python main.py all")
    
    print("\nDataset:")
    data_file = Path("cleaned_augmented_data.csv")
    if data_file.exists():
        import pandas as pd
        df = pd.read_csv(data_file)
        print(f"   Total samples: {len(df)}")
        print("   Class distribution:")
        class_counts = df['disease_clean'].value_counts()
        for class_name, count in class_counts.items():
            print(f"     {class_name}: {count}")
    
    print("\nProject structure:")
    print("   A2/")
    print("   ├── data.yaml              # YOLO config")
    print("   ├── train_yolov8.py        # training")
    print("   ├── inference.py           # inference")
    print("   ├── visualize_results.py   # visualization")
    print("   ├── main.py                # orchestrator")
    print("   ├── test_environment.py    # environment check")
    print("   ├── requirements.txt       # dependencies")
    print("   ├── README.md              # documentation")
    print("   └── outputs/")
    print("       ├── models/")
    print("       ├── results/")
    print("       └── yolo_dataset/")
    
    print("\n🚀 快速开始:")
    print("1. 检查环境: python test_environment.py")
    print("2. 开始训练: python main.py train")
    print("3. 查看结果: python main.py visualize")

def demo_with_sample_data():
    """Quick sample dataset overview."""
    print("\nSample data:")
    
    # Check test images
    test_dir = Path("outputs/yolo_dataset/test/images")
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg"))
        if test_images:
            print(f"Found {len(test_images)} test images")
            print("First 5:")
            for i, img_path in enumerate(test_images[:5]):
                print(f"  {i+1}. {img_path.name}")
            
            # Basic label stats
            print("\nLabel stats (train subset):")
            yolo_dataset = Path("outputs/yolo_dataset")
            
            # Count labels (limit to first 100 files)
            train_labels = yolo_dataset / "train/labels"
            if train_labels.exists():
                label_files = list(train_labels.glob("*.txt"))
                class_counts = {i: 0 for i in range(5)}
                
                for label_file in label_files[:100]:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                if class_id in class_counts:
                                    class_counts[class_id] += 1
                
                class_names = ['Brown Spot', 'Leaf Scald', 'Rice Blast', 'Rice Tungro', 'Sheath Blight']
                for class_id, count in class_counts.items():
                    print(f"  {class_names[class_id]}: {count} labels")
        else:
            print("Test images directory is empty")
    else:
        print("Test images directory not found")

def main():
    """CLI entrypoint."""
    demo_without_model()
    demo_with_sample_data()
    
    print("\n" + "=" * 50)
    print("Next steps:")
    print("1) Check environment: python test_environment.py")
    print("2) Start training:  python main.py train")
    print("3) Read docs:       README.md")

if __name__ == "__main__":
    main()
