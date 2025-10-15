#!/usr/bin/env python3
"""
Data Augmentation Testing Tool
测试和可视化不同数据增强策略的效果

用途:
1. 可视化数据增强效果
2. 对比不同增强参数
3. 帮助选择最佳增强策略
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from PIL import Image
import random


class AugmentationTester:
    """数据增强测试器"""

    def __init__(self):
        self.test_images = []

    def load_sample_images(self, image_dir, num_samples=6):
        """加载样本图像"""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg"))

        if len(image_files) == 0:
            print(f"❌ 未找到图像: {image_dir}")
            return False

        # 随机选择样本
        self.test_images = random.sample(image_files, min(num_samples, len(image_files)))
        print(f"✓ 加载了 {len(self.test_images)} 张测试图像")
        return True

    def create_augmentation_pipeline(self, preset='balanced'):
        """
        创建数据增强pipeline

        预设:
        - conservative: 保守的增强
        - balanced: 平衡的增强 (推荐)
        - aggressive: 激进的增强
        """
        if preset == 'conservative':
            # 保守增强 - 适合高质量数据集
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.3
                ),
            ])

        elif preset == 'aggressive':
            # 激进增强 - 适合小数据集
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=40,
                    val_shift_limit=30,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=5),
                ], p=0.3),
                A.RandomScale(scale_limit=0.2, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=20,
                    p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    p=0.2
                ),
            ])

        else:  # balanced (推荐)
            # 平衡增强
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.6),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.4
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.4
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0)),
                    A.GaussianBlur(blur_limit=3),
                ], p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=0.4
                ),
                A.CoarseDropout(
                    max_holes=5,
                    max_height=24,
                    max_width=24,
                    p=0.15
                ),
            ])

        return transform

    def visualize_augmentation(self, preset='balanced', num_variants=4, save_path='outputs/augmentation_test.png'):
        """可视化数据增强效果"""
        if len(self.test_images) == 0:
            print("❌ 未加载测试图像")
            return

        transform = self.create_augmentation_pipeline(preset)

        num_images = len(self.test_images)
        fig, axes = plt.subplots(num_images, num_variants + 1, figsize=(20, 4 * num_images))

        if num_images == 1:
            axes = axes.reshape(1, -1)

        for i, image_path in enumerate(self.test_images):
            # 读取原始图像
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 显示原始图像
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'原始 / Original\n{image_path.name}', fontsize=10)
            axes[i, 0].axis('off')

            # 显示增强变体
            for j in range(num_variants):
                augmented = transform(image=image)['image']
                axes[i, j + 1].imshow(augmented)
                axes[i, j + 1].set_title(f'增强 {j + 1} / Aug {j + 1}', fontsize=10)
                axes[i, j + 1].axis('off')

        plt.suptitle(f'数据增强效果预览 - {preset.capitalize()} Preset', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 增强效果图已保存: {save_path}")

        plt.show()

    def compare_presets(self, save_path='outputs/augmentation_comparison.png'):
        """对比不同预设的增强效果"""
        if len(self.test_images) == 0:
            print("❌ 未加载测试图像")
            return

        presets = ['conservative', 'balanced', 'aggressive']
        num_images = min(3, len(self.test_images))

        fig, axes = plt.subplots(num_images, len(presets) + 1, figsize=(20, 5 * num_images))

        if num_images == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_images):
            image_path = self.test_images[i]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 原始图像
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'原始\n{image_path.name}', fontsize=10)
            axes[i, 0].axis('off')

            # 不同预设
            for j, preset in enumerate(presets):
                transform = self.create_augmentation_pipeline(preset)
                augmented = transform(image=image)['image']
                axes[i, j + 1].imshow(augmented)
                axes[i, j + 1].set_title(f'{preset.capitalize()}', fontsize=10)
                axes[i, j + 1].axis('off')

        plt.suptitle('数据增强预设对比 / Augmentation Presets Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 预设对比图已保存: {save_path}")

        plt.show()

    def analyze_augmentation_statistics(self, preset='balanced', num_samples=100):
        """分析数据增强的统计特性"""
        if len(self.test_images) == 0:
            print("❌ 未加载测试图像")
            return

        transform = self.create_augmentation_pipeline(preset)

        # 选择一张图像进行分析
        image_path = self.test_images[0]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 生成多个增强样本
        brightness_values = []
        contrast_values = []

        for _ in range(num_samples):
            augmented = transform(image=image)['image']
            brightness_values.append(np.mean(augmented))
            contrast_values.append(np.std(augmented))

        # 可视化统计
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(brightness_values, bins=20, color='skyblue', edgecolor='black')
        axes[0].set_title('亮度分布 / Brightness Distribution')
        axes[0].set_xlabel('平均亮度 / Mean Brightness')
        axes[0].set_ylabel('频次 / Frequency')
        axes[0].axvline(np.mean(image), color='red', linestyle='--', label='原始 / Original')
        axes[0].legend()

        axes[1].hist(contrast_values, bins=20, color='lightcoral', edgecolor='black')
        axes[1].set_title('对比度分布 / Contrast Distribution')
        axes[1].set_xlabel('标准差 / Std Dev')
        axes[1].set_ylabel('频次 / Frequency')
        axes[1].axvline(np.std(image), color='red', linestyle='--', label='原始 / Original')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('outputs/augmentation_statistics.png', dpi=150, bbox_inches='tight')
        print(f"✓ 统计分析图已保存: outputs/augmentation_statistics.png")
        plt.show()

        print(f"\n统计摘要 ({num_samples} 个样本):")
        print(f"  原始亮度: {np.mean(image):.2f}")
        print(f"  增强后亮度范围: {np.min(brightness_values):.2f} - {np.max(brightness_values):.2f}")
        print(f"  原始对比度: {np.std(image):.2f}")
        print(f"  增强后对比度范围: {np.min(contrast_values):.2f} - {np.max(contrast_values):.2f}")


def main():
    """主函数"""
    print("=" * 70)
    print("数据增强测试工具 / Data Augmentation Testing Tool")
    print("=" * 70)

    # 创建测试器
    tester = AugmentationTester()

    # 加载测试图像
    test_dir = "yolo_dataset/train/images"
    if not Path(test_dir).exists():
        print(f"❌ 测试目录不存在: {test_dir}")
        print("请先运行 main.py --mode train 生成数据集")
        return

    tester.load_sample_images(test_dir, num_samples=6)

    # 1. 可视化平衡预设的增强效果
    print("\n1. 生成平衡预设的增强效果...")
    tester.visualize_augmentation(preset='balanced', num_variants=4)

    # 2. 对比不同预设
    print("\n2. 对比不同增强预设...")
    tester.compare_presets()

    # 3. 统计分析
    print("\n3. 分析增强的统计特性...")
    tester.analyze_augmentation_statistics(preset='balanced', num_samples=100)

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

    print("\n建议:")
    print("  - 如果数据集质量好、数量足够: 使用 conservative 预设")
    print("  - 如果数据集中等: 使用 balanced 预设 (推荐)")
    print("  - 如果数据集很小: 使用 aggressive 预设")


if __name__ == "__main__":
    # 检查 albumentations
    try:
        import albumentations
    except ImportError:
        print("❌ 未安装 albumentations 库")
        print("安装命令: pip install albumentations")
        exit(1)

    main()
