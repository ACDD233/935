#!/usr/bin/env python3
"""
测试 Grad-CAM 修复的脚本
"""
import os
import sys
from pathlib import Path

def test_gradcam_fix():
    """测试 Grad-CAM 修复"""
    print("=" * 60)
    print("测试 Grad-CAM 修复")
    print("=" * 60)
    
    # 检查模型文件
    model_path = "Dhan-Shomadhan/1343258/models/fold_1_best.pt"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    # 检查测试图像
    test_image = "Field Background/Browon Spot/bs_wb_55.jpg"
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return False
    
    print(f"✅ 模型文件存在: {model_path}")
    print(f"✅ 测试图像存在: {test_image}")
    
    try:
        # 测试 pytorch-grad-cam 导入
        from pytorch_grad_cam import GradCAM
        print("✅ pytorch-grad-cam 导入成功")
    except ImportError as e:
        print(f"❌ pytorch-grad-cam 导入失败: {e}")
        print("请运行: pip install grad-cam")
        return False
    
    try:
        # 测试自定义 YOLO Grad-CAM
        from yolo_gradcam import YOLOGradCAM
        print("✅ 自定义 YOLO Grad-CAM 导入成功")
    except ImportError as e:
        print(f"❌ 自定义 YOLO Grad-CAM 导入失败: {e}")
        return False
    
    print("\n🎉 所有组件都准备就绪!")
    print("\n现在可以运行:")
    print("python main.py --mode visualize")
    
    return True

if __name__ == '__main__':
    test_gradcam_fix()
