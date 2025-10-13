#!/usr/bin/env python3
"""
YOLOv8 Rice Disease Detection Main Script
水稻病害检测主控制脚本
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


def check_dataset():
    """检查数据集"""
    print("📁 检查数据集...")
    
    yolo_dataset = Path("outputs/yolo_dataset")
    if not yolo_dataset.exists():
        print(f"❌ YOLO数据集目录不存在: {yolo_dataset}")
        return False
    
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']
    for dir_path in required_dirs:
        full_path = yolo_dataset / dir_path
        if not full_path.exists():
            print(f"❌ 数据集子目录不存在: {full_path}")
            return False
    
    # 检查数据文件
    data_yaml = Path("data.yaml")
    if not data_yaml.exists():
        print(f"❌ 配置文件不存在: {data_yaml}")
        return False
    
    print("✅ 数据集检查通过")
    return True

def train_model(args):
    """训练模型"""
    print("🏋️ 开始训练模型...")
    
    if not check_environment():
        return False
    
    if not check_dataset():
        return False
    
    try:
        # 运行训练脚本
        result = subprocess.run([sys.executable, "train_yolov8.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 模型训练完成!")
            print(result.stdout)
            return True
        else:
            print("❌ 模型训练失败!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {str(e)}")
        return False

def evaluate_model(args):
    """评估模型"""
    print("📊 评估模型...")
    
    model_path = "outputs/models/rice_disease_detection/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型")
        return False
    
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO(model_path)
        
        # 评估模型
        metrics = model.val(data="data.yaml")
        
        print("📈 模型性能指标:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {str(e)}")
        return False

def run_inference(args):
    """运行推理"""
    print("🔍 运行推理...")
    
    model_path = "outputs/models/rice_disease_detection/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型")
        return False
    
    if not args.input:
        print("❌ 请指定输入图片或目录")
        return False
    
    try:
        from inference import RiceDiseaseDetector
        
        # 创建检测器
        detector = RiceDiseaseDetector(model_path, confidence_threshold=args.confidence)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 单张图片
            print(f"检测图片: {input_path}")
            detections, annotated_image = detector.detect_image(input_path)
            
            print(f"检测到 {len(detections)} 个病害区域:")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class_name']} (置信度: {det['confidence']:.3f})")
                
        elif input_path.is_dir():
            # 批量检测
            print(f"批量检测目录: {input_path}")
            detector.detect_batch(input_path)
            
        else:
            print(f"❌ 输入路径不存在: {input_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 推理过程中出现错误: {str(e)}")
        return False

def visualize_results(args):
    """可视化结果"""
    print("📊 可视化结果...")
    
    model_path = "outputs/models/rice_disease_detection/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型")
        return False
    
    try:
        # 运行可视化脚本
        result = subprocess.run([sys.executable, "visualize_results.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 可视化完成!")
            print(result.stdout)
            return True
        else:
            print("❌ 可视化失败!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOv8 水稻病害检测系统")
    parser.add_argument("command", choices=["train", "eval", "infer", "visualize", "all"],
                       help="执行的操作: train(训练), eval(评估), infer(推理), visualize(可视化), all(全部)")
    parser.add_argument("--input", "-i", help="推理时的输入图片或目录")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="推理时的置信度阈值 (默认: 0.5)")
    
    args = parser.parse_args()
    
    print("🌾 YOLOv8 水稻病害检测系统")
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
        print("🚀 执行完整流程...")
        
        # 1. 训练模型
        print("\n1️⃣ 训练模型")
        if not train_model(args):
            success = False
            print("❌ 训练失败，停止后续流程")
            return
        
        # 2. 评估模型
        print("\n2️⃣ 评估模型")
        if not evaluate_model(args):
            success = False
        
        # 3. 可视化结果
        print("\n3️⃣ 可视化结果")
        if not visualize_results(args):
            success = False
        
        # 4. 示例推理
        print("\n4️⃣ 示例推理")
        test_dir = "outputs/yolo_dataset/test/images"
        if os.path.exists(test_dir):
            args.input = test_dir
            if not run_inference(args):
                success = False
        else:
            print(f"⚠️ 测试图片目录不存在: {test_dir}")
    
    if success:
        print("\n🎉 所有操作完成!")
        print("📁 检查以下目录查看结果:")
        print("  - 模型权重: outputs/models/rice_disease_detection/weights/")
        print("  - 训练结果: outputs/models/rice_disease_detection/")
        print("  - 可视化结果: outputs/results/")
    else:
        print("\n❌ 部分操作失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
