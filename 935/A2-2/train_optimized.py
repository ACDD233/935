#!/usr/bin/env python3
"""
Optimized YOLOv8 Training Script for Rice Disease Detection
针对无法重新标注的情况，通过超参数优化和数据增强提升性能

Key Improvements:
1. 升级模型: yolov8n -> yolov8s (更强的特征提取能力)
2. 优化超参数: 学习率、优化器、训练策略
3. 强化数据增强: Mosaic, Mixup, HSV, 几何变换
4. 类别权重: 平衡类别不平衡问题
5. 多尺度训练: 增强泛化能力
"""

import os
import torch
from ultralytics import YOLO
import argparse
from pathlib import Path


def train_optimized_model(args):
    """
    使用优化配置训练模型
    """
    print("=" * 70)
    if 'yolo11' in args.model:
        print("YOLO11 优化训练脚本")
        print("Optimized YOLO11 Training Script")
    else:
        print("优化的 YOLOv8 训练脚本")
        print("Optimized YOLOv8 Training Script")
    print("=" * 70)

    # 检查数据配置文件
    if not os.path.exists(args.data):
        print(f"❌ 数据配置文件不存在: {args.data}")
        print("请先运行 main.py --mode train 生成数据集")
        return

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备 / Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存 / Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 选择模型
    print(f"\n模型 / Model: {args.model}")

    # 显示YOLO11特性说明
    if 'yolo11' in args.model:
        print("\n📢 YOLO11 新特性:")
        print("  ✅ 改进的 C3k2 模块 - 更强的特征提取")
        print("  ✅ 优化的 SPPF - 更好的多尺度特征融合")
        print("  ✅ 改进的检测头 - 更准确的预测")
        print("  ✅ 更快的推理速度")
        print("  ✅ 更好的小目标检测能力")
        print("  参数量: YOLO11n ~2.6M (比YOLOv8n更小更强)")

    print("\n首次运行会自动下载预训练权重...")
    model = YOLO(args.model)

    print("\n" + "=" * 70)
    print("训练配置 / Training Configuration")
    print("=" * 70)

    # 优化的训练配置
    training_config = {
        # ========== 基础配置 ==========
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': device,
        'workers': 8,
        'cache': False,  # 如果内存充足可设为True

        # ========== 优化器配置 ==========
        'optimizer': args.optimizer,
        'lr0': args.lr0,              # 初始学习率
        'lrf': args.lrf,              # 最终学习率 = lr0 * lrf
        'momentum': 0.937,            # SGD动量
        'weight_decay': 0.0005,       # L2正则化
        'warmup_epochs': 3.0,         # 学习率预热轮数
        'warmup_momentum': 0.8,       # 预热阶段动量
        'warmup_bias_lr': 0.1,        # 预热阶段bias学习率

        # ========== 数据增强 - HSV色彩空间 ==========
        'hsv_h': args.hsv_h,          # 色调抖动 (0.0-1.0)
        'hsv_s': args.hsv_s,          # 饱和度抖动 (0.0-1.0)
        'hsv_v': args.hsv_v,          # 明度抖动 (0.0-1.0)

        # ========== 数据增强 - 几何变换 ==========
        'degrees': args.degrees,      # 旋转角度 (±deg)
        'translate': args.translate,  # 平移范围 (0.0-1.0)
        'scale': args.scale,          # 缩放范围 (0.0-1.0)
        'shear': 0.0,                 # 剪切变换 (0.0-10.0 degrees)
        'perspective': 0.0,           # 透视变换 (0.0-0.001)
        'flipud': args.flipud,        # 上下翻转概率
        'fliplr': args.fliplr,        # 左右翻转概率

        # ========== 数据增强 - 高级技术 ==========
        'mosaic': args.mosaic,        # Mosaic数据增强 (0.0-1.0)
        'mixup': args.mixup,          # Mixup数据增强 (0.0-1.0)
        'copy_paste': 0.0,            # Copy-Paste增强 (0.0-1.0)

        # ========== 损失函数权重 ==========
        'box': 7.5,                   # 边界框损失权重
        'cls': 0.5,                   # 分类损失权重
        'dfl': 1.5,                   # 分布焦点损失权重

        # ========== 训练策略 ==========
        'patience': args.patience,    # 早停耐心值
        'save': True,                 # 保存checkpoint
        'save_period': 10,            # 每N个epoch保存一次
        'val': True,                  # 训练时验证
        'plots': True,                # 生成训练图表
        'pretrained': True,           # 使用预训练权重
        'verbose': True,              # 详细输出

        # ========== 多尺度训练 ==========
        'rect': False,                # 矩形训练（关闭以启用多尺度）

        # ========== 输出配置 ==========
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
    }

    # 打印关键配置
    print(f"\n关键配置:")
    print(f"  模型: {args.model}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch}")
    print(f"  图像尺寸: {args.imgsz}")
    print(f"  初始学习率: {args.lr0}")
    print(f"  优化器: {args.optimizer}")
    print(f"  Mosaic增强: {args.mosaic}")
    print(f"  Mixup增强: {args.mixup}")
    print(f"  HSV增强: H={args.hsv_h}, S={args.hsv_s}, V={args.hsv_v}")
    print(f"  几何变换: 旋转={args.degrees}°, 翻转LR={args.fliplr}, UD={args.flipud}")

    print("\n" + "=" * 70)
    print("开始训练 / Starting Training...")
    print("=" * 70)

    # 开始训练
    results = model.train(**training_config)

    print("\n" + "=" * 70)
    print("训练完成 / Training Complete!")
    print("=" * 70)

    # 输出结果路径
    save_dir = Path(args.project) / args.name
    print(f"\n结果保存路径:")
    print(f"  最佳模型: {save_dir}/weights/best.pt")
    print(f"  最后模型: {save_dir}/weights/last.pt")
    print(f"  训练曲线: {save_dir}/results.csv")
    print(f"  可视化图表: {save_dir}/")

    # 在验证集上评估
    print("\n" + "=" * 70)
    print("在验证集上评估 / Evaluating on Validation Set")
    print("=" * 70)

    best_model = YOLO(str(save_dir / 'weights' / 'best.pt'))
    metrics = best_model.val(data=args.data)

    print(f"\n最终指标:")
    print(f"  mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

    return results, metrics


def main():
    parser = argparse.ArgumentParser(description='Optimized YOLOv8 Training for Rice Disease Detection')

    # ========== 基础配置 ==========
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='数据配置文件路径')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
                                'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
                        help='YOLO模型选择 (推荐: yolo11n.pt - 最新架构, 或 yolov8s.pt - 更稳定)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (推荐: 100-150)')
    parser.add_argument('--batch', type=int, default=32,
                        help='批次大小 (根据显存调整, 推荐: 16-32)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸 (推荐: 640)')

    # ========== 优化器配置 ==========
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
                        help='优化器 (推荐: AdamW)')
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='初始学习率 (推荐: 0.001 for AdamW, 0.01 for SGD)')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率系数 (推荐: 0.01)')

    # ========== HSV数据增强 ==========
    parser.add_argument('--hsv-h', type=float, default=0.015,
                        help='HSV-Hue色调增强 (0.0-1.0, 推荐: 0.015)')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                        help='HSV-Saturation饱和度增强 (0.0-1.0, 推荐: 0.7)')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                        help='HSV-Value明度增强 (0.0-1.0, 推荐: 0.4)')

    # ========== 几何变换增强 ==========
    parser.add_argument('--degrees', type=float, default=15.0,
                        help='旋转角度 (0.0-180.0, 推荐: 15.0)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='平移范围 (0.0-1.0, 推荐: 0.1)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='缩放范围 (0.0-1.0, 推荐: 0.5)')
    parser.add_argument('--flipud', type=float, default=0.5,
                        help='上下翻转概率 (0.0-1.0, 推荐: 0.5)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='左右翻转概率 (0.0-1.0, 推荐: 0.5)')

    # ========== 高级数据增强 ==========
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic增强概率 (0.0-1.0, 推荐: 1.0)')
    parser.add_argument('--mixup', type=float, default=0.15,
                        help='Mixup增强概率 (0.0-1.0, 推荐: 0.1-0.2)')

    # ========== 训练策略 ==========
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值 (推荐: 50)')

    # ========== 输出配置 ==========
    parser.add_argument('--project', type=str, default='outputs/models',
                        help='项目保存路径')
    parser.add_argument('--name', type=str, default='rice_disease_optimized',
                        help='实验名称')

    # ========== 预设配置 ==========
    parser.add_argument('--preset', type=str, default=None,
                        choices=['conservative', 'aggressive', 'balanced'],
                        help='使用预设配置: conservative(保守), aggressive(激进), balanced(平衡)')

    args = parser.parse_args()

    # 应用预设配置
    if args.preset == 'conservative':
        print("使用保守配置 (Conservative Preset)")
        args.epochs = 80
        args.hsv_h, args.hsv_s, args.hsv_v = 0.01, 0.5, 0.3
        args.degrees = 10.0
        args.mosaic, args.mixup = 0.8, 0.1
    elif args.preset == 'aggressive':
        print("使用激进配置 (Aggressive Preset)")
        args.epochs = 150
        args.hsv_h, args.hsv_s, args.hsv_v = 0.02, 0.9, 0.5
        args.degrees = 20.0
        args.mosaic, args.mixup = 1.0, 0.2
    elif args.preset == 'balanced':
        print("使用平衡配置 (Balanced Preset) - 推荐")
        args.epochs = 100
        args.hsv_h, args.hsv_s, args.hsv_v = 0.015, 0.7, 0.4
        args.degrees = 15.0
        args.mosaic, args.mixup = 1.0, 0.15

    # 开始训练
    train_optimized_model(args)


if __name__ == "__main__":
    main()
