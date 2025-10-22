#!/usr/bin/env python3
"""
测试设备匹配修复
"""
import torch
import numpy as np
from ultralytics import YOLO

def test_device_matching():
    """测试设备匹配修复"""
    print("=" * 60)
    print("测试设备匹配修复")
    print("=" * 60)
    
    # 1. 测试模型加载和设备设置
    model_path = "Dhan-Shomadhan/1343258/models/fold_1_best.pt"
    try:
        yolo_model = YOLO(model_path)
        torch_model = yolo_model.model
        torch_model.eval()
        
        # 确保模型在正确的设备上
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch_model = torch_model.to(device)
        
        model_device = next(torch_model.parameters()).device
        print(f"✅ 模型设备: {model_device}")
        print(f"✅ 目标设备: {device}")
        print(f"✅ 设备匹配: {model_device == torch.device(device)}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 2. 测试模型包装器
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.model.train()
            # 确保模型在正确的设备上
            self.device = next(model.parameters()).device
            self.model = self.model.to(self.device)
            print(f"✅ 包装器设备: {self.device}")
        
        def forward(self, x):
            # 确保输入在正确的设备上
            if x.device != self.device:
                x = x.to(self.device)
                print(f"✅ 输入设备调整: {x.device}")
            
            if not x.requires_grad:
                x = x.requires_grad_(True)
            
            with torch.enable_grad():
                output = self.model(x)
            
            if isinstance(output, tuple):
                return output[0]
            return output
    
    try:
        wrapped_model = ModelWrapper(torch_model)
        print("✅ 模型包装器创建成功")
    except Exception as e:
        print(f"❌ 模型包装器创建失败: {e}")
        return False
    
    # 3. 测试输入张量
    try:
        test_input = torch.randn(1, 3, 320, 320).to(device)
        print(f"✅ 输入张量设备: {test_input.device}")
        print(f"✅ 输入张量类型: {test_input.dtype}")
        print(f"✅ 输入张量形状: {test_input.shape}")
        
        # 启用梯度
        test_input.requires_grad_(True)
        print(f"✅ 输入张量梯度: {test_input.requires_grad}")
        
    except Exception as e:
        print(f"❌ 输入张量创建失败: {e}")
        return False
    
    # 4. 测试前向传播
    try:
        with torch.enable_grad():
            output = wrapped_model(test_input)
            print(f"✅ 前向传播成功")
            print(f"✅ 输出类型: {type(output)}")
            if hasattr(output, 'shape'):
                print(f"✅ 输出形状: {output.shape}")
            if hasattr(output, 'device'):
                print(f"✅ 输出设备: {output.device}")
                
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False
    
    # 5. 测试 pytorch-grad-cam 导入
    try:
        from pytorch_grad_cam import GradCAM
        print("✅ pytorch-grad-cam 导入成功")
    except ImportError as e:
        print(f"❌ pytorch-grad-cam 导入失败: {e}")
        return False
    
    print("\n🎉 设备匹配测试通过!")
    return True

if __name__ == '__main__':
    test_device_matching()
