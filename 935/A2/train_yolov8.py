#!/usr/bin/env python3
"""
YOLOv8 Rice Disease Detection Training Script
训练YOLOv8模型用于水稻病害检测
"""

import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def setup_environment():
    """设置训练环境"""
    print("🚀 设置YOLOv8训练环境...")
    
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return device, output_dir

def validate_dataset(data_yaml_path):
    """验证数据集配置"""
    print("🔍 验证数据集配置...")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"配置文件不存在: {data_yaml_path}")
    
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 检查数据集路径
    base_path = Path(data_config['path'])
    train_path = base_path / data_config['train']
    val_path = base_path / data_config['val']
    test_path = base_path / data_config['test']
    
    print(f"📂 数据集根目录: {base_path}")
    print(f"📂 训练集: {train_path}")
    print(f"📂 验证集: {val_path}")
    print(f"📂 测试集: {test_path}")
    
    # 检查图片数量
    train_images = len(list(train_path.glob('*.jpg')))
    val_images = len(list(val_path.glob('*.jpg')))
    test_images = len(list(test_path.glob('*.jpg')))
    
    print(f"📊 训练图片数量: {train_images}")
    print(f"📊 验证图片数量: {val_images}")
    print(f"📊 测试图片数量: {test_images}")
    print(f"📊 类别数量: {data_config['nc']}")
    print(f"📊 类别名称: {data_config['names']}")
    
    return data_config

def train_model(data_yaml_path, device, output_dir, epochs=100):
    """训练YOLOv8模型"""
    print("🏋️ 开始训练YOLOv8模型...")
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用nano版本，也可以选择s, m, l, x
    
    # 训练参数
    train_args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'project': str(output_dir),
        'name': 'rice_disease_detection',
        'save': True,
        'save_period': 10,
        'cache': False,
        'workers': 8,
        'patience': 20,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
        'val': True,
        'plots': True,
        'verbose': True
    }
    
    print("📋 训练参数:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # 开始训练
    results = model.train(**train_args)
    
    print("✅ 训练完成!")
    return results, model

def evaluate_model(model, data_yaml_path):
    """评估模型性能"""
    print("📊 评估模型性能...")
    
    # 在验证集上评估
    metrics = model.val(data=data_yaml_path)
    
    print("📈 模型性能指标:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics

def main():
    """主函数"""
    print("🌾 YOLOv8 水稻病害检测模型训练")
    print("=" * 50)
    
    # 设置环境
    device, output_dir = setup_environment()
    
    # 数据配置文件路径
    data_yaml_path = "data.yaml"
    
    try:
        # 验证数据集
        data_config = validate_dataset(data_yaml_path)
        
        # 训练模型
        results, model = train_model(data_yaml_path, device, output_dir, epochs=100)
        
        # 评估模型
        metrics = evaluate_model(model, data_yaml_path)
        
        # 保存最佳模型
        best_model_path = output_dir / "rice_disease_detection" / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"💾 最佳模型已保存到: {best_model_path}")
        
        print("\n🎉 训练流程完成!")
        print("📁 检查以下目录查看结果:")
        print(f"  - 训练结果: {output_dir / 'rice_disease_detection'}")
        print(f"  - 模型权重: {output_dir / 'rice_disease_detection' / 'weights'}")
        print(f"  - 训练图表: {output_dir / 'rice_disease_detection' / 'results.png'}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
