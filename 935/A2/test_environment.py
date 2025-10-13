#!/usr/bin/env python3
"""
环境测试脚本 - 验证YOLOv8环境是否正常
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试必要的包导入"""
    print("🔍 测试包导入...")
    
    try:
        import ultralytics
        print(f"✅ ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"❌ ultralytics导入失败: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA设备: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ torch导入失败: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ opencv: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ opencv导入失败: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ matplotlib导入失败: {e}")
        return False
    
    return True

def test_yolo_model():
    """测试YOLO模型加载"""
    print("\n🤖 测试YOLO模型加载...")
    
    try:
        from ultralytics import YOLO
        
        # 尝试加载预训练模型
        print("加载YOLOv8n预训练模型...")
        model = YOLO('yolov8n.pt')
        print("✅ YOLO模型加载成功")
        
        return True
    except Exception as e:
        print(f"❌ YOLO模型加载失败: {e}")
        return False

def test_dataset():
    """测试数据集结构"""
    print("\n📁 测试数据集结构...")
    
    # 检查data.yaml
    data_yaml = Path("data.yaml")
    if data_yaml.exists():
        print("✅ data.yaml存在")
    else:
        print("❌ data.yaml不存在")
        return False
    
    # 检查YOLO数据集目录
    yolo_dir = Path("outputs/yolo_dataset")
    if yolo_dir.exists():
        print("✅ YOLO数据集目录存在")
        
        # 检查子目录
        required_dirs = ['train/images', 'val/images', 'test/images']
        for dir_name in required_dirs:
            dir_path = yolo_dir / dir_name
            if dir_path.exists():
                image_count = len(list(dir_path.glob('*.jpg')))
                print(f"✅ {dir_name}: {image_count} 张图片")
            else:
                print(f"❌ {dir_name}不存在")
                return False
    else:
        print("❌ YOLO数据集目录不存在")
        return False
    
    return True

def test_simple_inference():
    """测试简单推理"""
    print("\n🔍 测试简单推理...")
    
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO('yolov8n.pt')
        
        # 创建一个简单的测试图片路径
        test_dir = Path("outputs/yolo_dataset/test/images")
        if test_dir.exists():
            test_images = list(test_dir.glob("*.jpg"))
            if test_images:
                test_image = test_images[0]
                print(f"使用测试图片: {test_image.name}")
                
                # 运行推理
                results = model(str(test_image))
                print("✅ 推理测试成功")
                return True
            else:
                print("❌ 测试图片目录为空")
                return False
        else:
            print("❌ 测试图片目录不存在")
            return False
            
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 YOLOv8 环境测试")
    print("=" * 40)
    
    all_tests_passed = True
    
    # 1. 测试包导入
    if not test_imports():
        all_tests_passed = False
    
    # 2. 测试YOLO模型
    if not test_yolo_model():
        all_tests_passed = False
    
    # 3. 测试数据集
    if not test_dataset():
        all_tests_passed = False
    
    # 4. 测试推理
    if not test_simple_inference():
        all_tests_passed = False
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("🎉 所有测试通过! 环境配置正确")
        print("\n📋 接下来可以:")
        print("1. 运行 'python main.py train' 开始训练")
        print("2. 运行 'python main.py all' 执行完整流程")
    else:
        print("❌ 部分测试失败，请检查环境配置")
        print("\n🔧 可能的解决方案:")
        print("1. 安装缺失的依赖: pip install -r requirements.txt")
        print("2. 检查数据集路径是否正确")
        print("3. 确保有足够的磁盘空间和内存")

if __name__ == "__main__":
    main()
