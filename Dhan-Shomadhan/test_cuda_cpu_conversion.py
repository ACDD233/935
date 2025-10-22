#!/usr/bin/env python3
"""
测试 CUDA 到 CPU 转换修复
"""
import torch
import numpy as np
from ultralytics import YOLO

def test_cuda_to_cpu_conversion():
    """测试 CUDA 到 CPU 转换修复"""
    print("=" * 60)
    print("测试 CUDA 到 CPU 转换修复")
    print("=" * 60)
    
    # 1. 检查 CUDA 可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ CUDA 可用，使用设备: {device}")
    else:
        device = 'cpu'
        print(f"✅ 使用 CPU 设备: {device}")
    
    # 2. 加载模型
    model_path = "Dhan-Shomadhan/1343258/models/fold_1_best.pt"
    try:
        yolo_model = YOLO(model_path)
        torch_model = yolo_model.model
        torch_model.eval()
        
        # 确保模型在正确的设备上
        torch_model = torch_model.to(device)
        print("✅ YOLO 模型加载成功")
        
    except Exception as e:
        print(f"❌ YOLO 模型加载失败: {e}")
        return False
    
    # 3. 创建测试输入
    try:
        test_input = torch.randn(1, 3, 320, 320).to(device)
        print(f"✅ 测试输入创建成功，设备: {test_input.device}")
        
    except Exception as e:
        print(f"❌ 测试输入创建失败: {e}")
        return False
    
    # 4. 测试模型输出
    try:
        with torch.no_grad():
            outputs = torch_model(test_input)
            print(f"✅ 模型输出类型: {type(outputs)}")
            
            # 处理输出格式
            if isinstance(outputs, tuple):
                main_output = outputs[0]
                print(f"✅ 元组输出，主输出类型: {type(main_output)}")
            else:
                main_output = outputs
                print(f"✅ 单个输出类型: {type(main_output)}")
            
            # 测试 CUDA 到 CPU 转换
            if hasattr(main_output, 'probs'):
                print("✅ 使用 probs 属性")
                try:
                    # 正确的转换方式
                    pred_probs = main_output.probs.data.cpu().numpy()
                    pred_class = main_output.probs.top1
                    print(f"✅ CUDA 到 CPU 转换成功")
                    print(f"✅ 预测概率形状: {pred_probs.shape}")
                    print(f"✅ 预测类别: {pred_class}")
                except Exception as e:
                    print(f"❌ CUDA 到 CPU 转换失败: {e}")
                    return False
            else:
                if isinstance(main_output, torch.Tensor):
                    print("✅ 使用张量 softmax")
                    try:
                        # 正确的转换方式
                        pred_probs = torch.softmax(main_output, dim=1).detach().cpu().numpy()[0]
                        pred_class = main_output.argmax(dim=1).item()
                        print(f"✅ CUDA 到 CPU 转换成功")
                        print(f"✅ 预测概率形状: {pred_probs.shape}")
                        print(f"✅ 预测类别: {pred_class}")
                    except Exception as e:
                        print(f"❌ CUDA 到 CPU 转换失败: {e}")
                        return False
                else:
                    print("❌ 主输出既不是张量也没有 probs 属性")
                    return False
                    
    except Exception as e:
        print(f"❌ 模型输出测试失败: {e}")
        return False
    
    # 5. 测试错误的转换方式（应该失败）
    print("\n测试错误的转换方式:")
    try:
        test_tensor = torch.randn(1, 3, 320, 320).to(device)
        # 错误的转换方式（应该失败）
        wrong_result = test_tensor.numpy()
        print("❌ 错误转换方式不应该成功")
        return False
    except Exception as e:
        print(f"✅ 错误转换方式正确失败: {e}")
    
    print("\n🎉 CUDA 到 CPU 转换测试通过!")
    return True

if __name__ == '__main__':
    test_cuda_to_cpu_conversion()
