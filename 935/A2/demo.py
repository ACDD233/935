#!/usr/bin/env python3
"""
YOLOv8 水稻病害检测演示脚本
展示如何使用训练好的模型进行病害检测
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def demo_without_model():
    """演示脚本 - 不依赖训练好的模型"""
    print("🌾 YOLOv8 水稻病害检测系统演示")
    print("=" * 50)
    
    print("\n📋 系统功能:")
    print("1. 自动检测水稻叶片上的5种病害:")
    print("   - Brown Spot (褐斑病)")
    print("   - Leaf Scald (叶鞘腐败病)")  
    print("   - Rice Blast (稻瘟病)")
    print("   - Rice Tungro (东格鲁病毒病)")
    print("   - Sheath Blight (纹枯病)")
    
    print("\n🔧 使用方法:")
    print("1. 训练模型:")
    print("   python main.py train")
    print("   或")
    print("   python train_yolov8.py")
    
    print("\n2. 评估模型:")
    print("   python main.py eval")
    
    print("\n3. 检测图片:")
    print("   python main.py infer --input image.jpg")
    print("   python main.py infer --input image_directory/")
    
    print("\n4. 可视化结果:")
    print("   python main.py visualize")
    
    print("\n5. 完整流程:")
    print("   python main.py all")
    
    print("\n📊 数据集信息:")
    # 显示数据集统计
    data_file = Path("cleaned_augmented_data.csv")
    if data_file.exists():
        import pandas as pd
        df = pd.read_csv(data_file)
        print(f"   总样本数: {len(df)}")
        print("   各类别分布:")
        class_counts = df['disease_clean'].value_counts()
        for class_name, count in class_counts.items():
            print(f"     {class_name}: {count}")
    
    print("\n📁 项目结构:")
    print("   A2/")
    print("   ├── data.yaml              # YOLO配置文件")
    print("   ├── train_yolov8.py        # 训练脚本")
    print("   ├── inference.py           # 推理脚本")
    print("   ├── visualize_results.py   # 可视化脚本")
    print("   ├── main.py               # 主控制脚本")
    print("   ├── test_environment.py   # 环境测试脚本")
    print("   ├── requirements.txt      # 依赖包")
    print("   ├── README.md             # 项目说明")
    print("   └── outputs/              # 输出目录")
    print("       ├── models/           # 训练好的模型")
    print("       ├── results/          # 可视化结果")
    print("       └── yolo_dataset/     # YOLO格式数据集")
    
    print("\n🚀 快速开始:")
    print("1. 检查环境: python test_environment.py")
    print("2. 开始训练: python main.py train")
    print("3. 查看结果: python main.py visualize")

def demo_with_sample_data():
    """使用样本数据演示检测结果"""
    print("\n🖼️ 样本检测演示:")
    
    # 检查是否有测试图片
    test_dir = Path("outputs/yolo_dataset/test/images")
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg"))
        if test_images:
            print(f"找到 {len(test_images)} 张测试图片")
            print("前5张图片:")
            for i, img_path in enumerate(test_images[:5]):
                print(f"  {i+1}. {img_path.name}")
            
            # 显示数据集类别信息
            print("\n📊 数据集类别信息:")
            yolo_dataset = Path("outputs/yolo_dataset")
            
            # 统计训练集标签
            train_labels = yolo_dataset / "train/labels"
            if train_labels.exists():
                label_files = list(train_labels.glob("*.txt"))
                class_counts = {i: 0 for i in range(5)}
                
                for label_file in label_files[:100]:  # 只检查前100个文件
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                if class_id in class_counts:
                                    class_counts[class_id] += 1
                
                class_names = ['Brown Spot', 'Leaf Scald', 'Rice Blast', 'Rice Tungro', 'Sheath Blight']
                for class_id, count in class_counts.items():
                    print(f"  {class_names[class_id]}: {count} 个标注")
        else:
            print("测试图片目录为空")
    else:
        print("测试图片目录不存在")

def main():
    """主函数"""
    demo_without_model()
    demo_with_sample_data()
    
    print("\n" + "=" * 50)
    print("🎯 下一步操作建议:")
    print("1. 运行环境测试: python test_environment.py")
    print("2. 开始模型训练: python main.py train")
    print("3. 查看详细说明: cat README.md")

if __name__ == "__main__":
    main()
