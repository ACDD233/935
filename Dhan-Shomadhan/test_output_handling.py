#!/usr/bin/env python3
"""
测试 YOLO 模型输出格式处理
"""
import torch
import numpy as np
from ultralytics import YOLO

def test_yolo_output_handling():
    """测试 YOLO 模型输出格式处理"""
    print("=" * 60)
    print("测试 YOLO 模型输出格式处理")
    print("=" * 60)
    
    # 加载模型
    model_path = "Dhan-Shomadhan/1343258/models/fold_1_best.pt"
    try:
        yolo_model = YOLO(model_path)
        torch_model = yolo_model.model
        torch_model.eval()
        
        # 确保模型在正确的设备上
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch_model = torch_model.to(device)
        print(f"✅ YOLO 模型加载成功，设备: {device}")
    except Exception as e:
        print(f"❌ YOLO 模型加载失败: {e}")
        return False
    
    # 创建测试输入
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_input = torch.randn(1, 3, 320, 320).to(device)
    
    # 测试模型输出
    try:
        with torch.no_grad():
            outputs = torch_model(test_input)
            print(f"✅ 模型输出类型: {type(outputs)}")
            
            if isinstance(outputs, tuple):
                print(f"✅ 输出是元组，包含 {len(outputs)} 个元素")
                for i, item in enumerate(outputs):
                    print(f"  元素 {i}: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"    Shape: {item.shape}")
                    if hasattr(item, 'probs'):
                        print(f"    有 probs 属性")
            else:
                print(f"✅ 输出是单个对象: {type(outputs)}")
                if hasattr(outputs, 'shape'):
                    print(f"  Shape: {outputs.shape}")
                if hasattr(outputs, 'probs'):
                    print(f"  有 probs 属性")
            
            # 测试输出处理逻辑
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs
            
            print(f"✅ 主输出类型: {type(main_output)}")
            
            if hasattr(main_output, 'probs'):
                print("✅ 使用 probs 属性获取预测")
                pred_probs = main_output.probs.data.numpy()
                pred_class = main_output.probs.top1
            else:
                if isinstance(main_output, torch.Tensor):
                    print("✅ 使用张量 softmax 获取预测")
                    pred_probs = torch.softmax(main_output, dim=1).detach().numpy()[0]
                    pred_class = main_output.argmax(dim=1).item()
                else:
                    print("❌ 主输出既不是张量也没有 probs 属性")
                    return False
            
            print(f"✅ 预测类别: {pred_class}")
            print(f"✅ 预测概率形状: {pred_probs.shape}")
            
    except Exception as e:
        print(f"❌ 输出处理失败: {e}")
        return False
    
    print("\n🎉 YOLO 输出格式处理测试通过!")
    return True

if __name__ == '__main__':
    test_yolo_output_handling()
