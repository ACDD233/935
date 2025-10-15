#!/usr/bin/env python3
"""
Hard Example Mining for Rice Disease Detection
困难样本挖掘工具

功能:
1. 找出低置信度检测样本
2. 找出完全漏检的样本
3. 找出误分类样本
4. 生成困难样本报告
5. 建议重训练策略
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class HardExampleMiner:
    """困难样本挖掘器"""

    def __init__(self, model_path, data_dir, class_names=None):
        """
        初始化

        Args:
            model_path: 模型路径
            data_dir: 测试图像目录
            class_names: 类别名称列表
        """
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.class_names = class_names or ['Brown Spot', 'Leaf Scald', 'Rice Blast', 'Rice Tungro', 'Sheath Blight']
        self.model = None

        # 结果存储
        self.hard_examples = {
            'no_detection': [],          # 漏检样本
            'low_confidence': [],        # 低置信度样本
            'misclassified': [],         # 误分类样本
            'multi_detection': [],       # 多目标检测样本（可能是误检）
        }

        # 加载模型
        self.load_model()

    def load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        print(f"加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("✓ 模型加载成功")

    def get_true_label_from_filename(self, filename):
        """从文件名推断真实标签"""
        name = filename.lower()

        # 根据文件名前缀判断类别
        if name.startswith('bsf_') or name.startswith('bs_wb_'):
            return 'Brown Spot'
        elif name.startswith('lsf_') or name.startswith('ls_wb_'):
            return 'Leaf Scald'
        elif name.startswith('rbf_') or name.startswith('rb_wb_'):
            return 'Rice Blast'
        elif name.startswith('rtf_') or name.startswith('rt_wb_'):
            return 'Rice Tungro'
        elif name.startswith('sbf_') or name.startswith('sb_wb_'):
            return 'Sheath Blight'
        else:
            return None

    def mine_hard_examples(self, confidence_threshold=0.7):
        """
        挖掘困难样本

        Args:
            confidence_threshold: 置信度阈值，低于此值视为困难样本
        """
        print("\n" + "=" * 70)
        print("开始挖掘困难样本...")
        print("=" * 70)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        # 获取所有图像
        image_files = list(self.data_dir.glob("*.jpg"))
        print(f"找到 {len(image_files)} 张图像")

        # 统计信息
        stats = {
            'total': len(image_files),
            'no_detection': 0,
            'low_confidence': 0,
            'misclassified': 0,
            'multi_detection': 0,
        }

        # 类别统计
        class_stats = defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'no_detection': 0,
            'low_confidence': 0,
            'avg_confidence': []
        })

        # 遍历所有图像
        for i, img_path in enumerate(image_files, 1):
            if i % 50 == 0:
                print(f"  处理进度: {i}/{len(image_files)}")

            filename = img_path.name
            true_label = self.get_true_label_from_filename(filename)

            if true_label is None:
                continue

            # 更新类别总数
            class_stats[true_label]['total'] += 1

            # 运行推理
            results = self.model(str(img_path), conf=0.5, verbose=False)

            # 分析结果
            if len(results[0].boxes) == 0:
                # 漏检
                self.hard_examples['no_detection'].append({
                    'image': filename,
                    'true_label': true_label,
                    'path': str(img_path)
                })
                stats['no_detection'] += 1
                class_stats[true_label]['no_detection'] += 1

            else:
                # 有检测结果
                boxes = results[0].boxes

                # 多目标检测
                if len(boxes) > 1:
                    self.hard_examples['multi_detection'].append({
                        'image': filename,
                        'true_label': true_label,
                        'num_detections': len(boxes),
                        'path': str(img_path)
                    })
                    stats['multi_detection'] += 1

                # 分析每个检测框
                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    pred_label = self.class_names[cls_id]

                    class_stats[true_label]['avg_confidence'].append(conf)

                    # 低置信度
                    if conf < confidence_threshold:
                        self.hard_examples['low_confidence'].append({
                            'image': filename,
                            'true_label': true_label,
                            'pred_label': pred_label,
                            'confidence': conf,
                            'path': str(img_path)
                        })
                        stats['low_confidence'] += 1
                        class_stats[true_label]['low_confidence'] += 1

                    # 误分类
                    if pred_label != true_label:
                        self.hard_examples['misclassified'].append({
                            'image': filename,
                            'true_label': true_label,
                            'pred_label': pred_label,
                            'confidence': conf,
                            'path': str(img_path)
                        })
                        stats['misclassified'] += 1
                        class_stats[true_label]['incorrect'] += 1
                    else:
                        class_stats[true_label]['correct'] += 1

        # 计算每类平均置信度
        for class_name in class_stats:
            if class_stats[class_name]['avg_confidence']:
                class_stats[class_name]['avg_confidence'] = np.mean(class_stats[class_name]['avg_confidence'])
            else:
                class_stats[class_name]['avg_confidence'] = 0.0

        print("\n" + "=" * 70)
        print("困难样本挖掘完成！")
        print("=" * 70)

        # 打印统计
        self.print_statistics(stats, class_stats)

        return stats, class_stats

    def print_statistics(self, stats, class_stats):
        """打印统计信息"""
        print(f"\n整体统计:")
        print(f"  总图像数: {stats['total']}")
        print(f"  漏检样本: {stats['no_detection']} ({stats['no_detection']/stats['total']*100:.1f}%)")
        print(f"  低置信度: {stats['low_confidence']} ({stats['low_confidence']/stats['total']*100:.1f}%)")
        print(f"  误分类: {stats['misclassified']} ({stats['misclassified']/stats['total']*100:.1f}%)")
        print(f"  多目标检测: {stats['multi_detection']} ({stats['multi_detection']/stats['total']*100:.1f}%)")

        print(f"\n各类别统计:")
        print(f"{'类别':<20} {'总数':<8} {'正确':<8} {'错误':<8} {'漏检':<8} {'低信':<8} {'平均置信度':<12}")
        print("-" * 100)
        for class_name in sorted(class_stats.keys()):
            cs = class_stats[class_name]
            print(f"{class_name:<20} {cs['total']:<8} {cs['correct']:<8} {cs['incorrect']:<8} "
                  f"{cs['no_detection']:<8} {cs['low_confidence']:<8} {cs['avg_confidence']:.4f}")

    def save_report(self, output_path='outputs/hard_examples_report.json'):
        """保存困难样本报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'model_path': self.model_path,
            'data_dir': str(self.data_dir),
            'hard_examples': self.hard_examples,
            'summary': {
                'no_detection_count': len(self.hard_examples['no_detection']),
                'low_confidence_count': len(self.hard_examples['low_confidence']),
                'misclassified_count': len(self.hard_examples['misclassified']),
                'multi_detection_count': len(self.hard_examples['multi_detection']),
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 报告已保存: {output_path}")

    def visualize_hard_examples(self, category='no_detection', max_samples=12, save_path=None):
        """
        可视化困难样本

        Args:
            category: 类别 ('no_detection', 'low_confidence', 'misclassified', 'multi_detection')
            max_samples: 最多显示的样本数
            save_path: 保存路径
        """
        if category not in self.hard_examples:
            print(f"❌ 无效类别: {category}")
            return

        samples = self.hard_examples[category][:max_samples]

        if len(samples) == 0:
            print(f"✓ 没有 {category} 类型的困难样本")
            return

        # 计算子图布局
        cols = 4
        rows = (len(samples) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(samples):
            row = i // cols
            col = i % cols

            # 读取图像
            img = cv2.imread(sample['path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[row, col].imshow(img)

            # 标题
            title = f"{sample['image']}\n真实: {sample['true_label']}"
            if 'pred_label' in sample:
                title += f"\n预测: {sample['pred_label']}"
            if 'confidence' in sample:
                title += f"\n置信度: {sample['confidence']:.3f}"

            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis('off')

        # 隐藏多余的子图
        for i in range(len(samples), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        category_names = {
            'no_detection': '漏检样本',
            'low_confidence': '低置信度样本',
            'misclassified': '误分类样本',
            'multi_detection': '多目标检测样本'
        }

        plt.suptitle(f'{category_names[category]} / {category.replace("_", " ").title()}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存
        if save_path is None:
            save_path = f'outputs/hard_examples_{category}.png'

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化已保存: {save_path}")

        plt.show()

    def generate_retraining_suggestions(self):
        """生成重训练建议"""
        print("\n" + "=" * 70)
        print("重训练建议 / Retraining Suggestions")
        print("=" * 70)

        # 分析最严重的问题
        issues = []

        if len(self.hard_examples['no_detection']) > 20:
            issues.append(("高漏检率", "增加数据增强强度，特别是Mosaic和Mixup"))

        if len(self.hard_examples['low_confidence']) > 30:
            issues.append(("低置信度样本多", "增加训练轮数，降低学习率"))

        if len(self.hard_examples['misclassified']) > 25:
            issues.append(("高误分类率", "使用更大的模型（yolov8s -> yolov8m），增加类别权重"))

        if len(self.hard_examples['multi_detection']) > 15:
            issues.append(("多目标检测", "调整NMS阈值，或检查数据标注"))

        if len(issues) == 0:
            print("✓ 模型性能良好，建议进行微调优化")
        else:
            print("\n发现的主要问题:")
            for i, (issue, suggestion) in enumerate(issues, 1):
                print(f"\n{i}. {issue}")
                print(f"   建议: {suggestion}")

        print("\n通用建议:")
        print("  1. 使用困难样本数据增强:")
        print("     - 将困难样本复制多份")
        print("     - 对困难样本应用更强的增强")
        print("  2. 调整训练策略:")
        print("     - 增加训练轮数至 100-150 epochs")
        print("     - 使用余弦学习率衰减")
        print("     - 增加 Mosaic 和 Mixup 概率")
        print("  3. 升级模型:")
        print("     - 从 yolov8n 升级到 yolov8s")
        print("     - 或尝试 yolov8m 获得更好性能")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Hard Example Mining for Rice Disease Detection')
    parser.add_argument('--model', type=str, default='outputs/models/rice_disease_detection3/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--data-dir', type=str, default='yolo_dataset/test/images',
                        help='测试图像目录')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='置信度阈值')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化困难样本')

    args = parser.parse_args()

    print("=" * 70)
    print("困难样本挖掘工具 / Hard Example Mining Tool")
    print("=" * 70)

    # 创建挖掘器
    miner = HardExampleMiner(args.model, args.data_dir)

    # 挖掘困难样本
    stats, class_stats = miner.mine_hard_examples(confidence_threshold=args.confidence_threshold)

    # 保存报告
    miner.save_report()

    # 可视化（可选）
    if args.visualize:
        print("\n生成可视化...")
        for category in ['no_detection', 'low_confidence', 'misclassified', 'multi_detection']:
            if len(miner.hard_examples[category]) > 0:
                miner.visualize_hard_examples(category, max_samples=12)

    # 生成建议
    miner.generate_retraining_suggestions()

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
