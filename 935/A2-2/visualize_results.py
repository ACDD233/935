#!/usr/bin/env python3
"""
YOLOv8 Rice Disease Detection Results Visualization
水稻病害检测结果可视化脚本
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
import cv2
from PIL import Image
import matplotlib.patches as patches
from ultralytics import YOLO

class ResultsVisualizer:
    """结果可视化器"""
    
    def __init__(self, results_dir="outputs/models/rice_disease_detection"):
        """
        初始化可视化器
        
        Args:
            results_dir: 训练结果目录
        """
        self.results_dir = Path(results_dir)
        self.class_names = ['Brown Spot', 'Leaf Scald', 'Rice Blast', 'Rice Tungro', 'Sheath Blight']
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
    def plot_training_curves(self, save_path="outputs/results/training_curves.png"):
        """绘制训练曲线"""
        print("📈 绘制训练曲线...")
        
        # 查找results.csv文件
        results_csv = self.results_dir / "results.csv"
        if not results_csv.exists():
            print(f"❌ 未找到训练结果文件: {results_csv}")
            return
        
        # 读取训练数据
        df = pd.read_csv(results_csv)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLOv8 水稻病害检测训练曲线', fontsize=16, fontweight='bold')
        
        # 1. 损失函数曲线
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='训练集 Box Loss', color='blue')
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='验证集 Box Loss', color='red')
        axes[0, 0].set_title('边界框损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 分类损失曲线
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='训练集 Cls Loss', color='blue')
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='验证集 Cls Loss', color='red')
        axes[0, 1].set_title('分类损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. mAP曲线
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
        axes[1, 0].set_title('平均精度 (mAP)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Precision和Recall曲线
        axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='purple')
        axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='brown')
        axes[1, 1].set_title('精确率和召回率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 训练曲线已保存到: {save_path}")
    
    def plot_confusion_matrix(self, model_path, data_yaml_path, save_path="outputs/results/confusion_matrix.png"):
        """绘制混淆矩阵"""
        print("📊 绘制混淆矩阵...")
        
        try:
            # 加载模型
            model = YOLO(model_path)
            
            # 在验证集上运行评估
            results = model.val(data=data_yaml_path, save_json=True)
            
            # 读取混淆矩阵数据
            conf_matrix_path = self.results_dir / "confusion_matrix.png"
            if conf_matrix_path.exists():
                # 复制现有的混淆矩阵
                import shutil
                shutil.copy2(conf_matrix_path, save_path)
                print(f"💾 混淆矩阵已保存到: {save_path}")
            else:
                print("⚠️ 未找到混淆矩阵文件")
                
        except Exception as e:
            print(f"❌ 生成混淆矩阵时出错: {str(e)}")
    
    def plot_class_distribution(self, data_csv_path="cleaned_augmented_data.csv", save_path="outputs/results/class_distribution.png"):
        """绘制类别分布图"""
        print("📊 绘制类别分布图...")
        
        if not os.path.exists(data_csv_path):
            print(f"❌ 数据文件不存在: {data_csv_path}")
            return
        
        # 读取数据
        df = pd.read_csv(data_csv_path)
        
        # 统计类别分布
        class_counts = df['disease_clean'].value_counts()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('水稻病害数据集类别分布', fontsize=16, fontweight='bold')
        
        # 柱状图
        bars = ax1.bar(range(len(class_counts)), class_counts.values, color=self.colors[:len(class_counts)])
        ax1.set_title('各类别样本数量')
        ax1.set_xlabel('病害类别')
        ax1.set_ylabel('样本数量')
        ax1.set_xticks(range(len(class_counts)))
        ax1.set_xticklabels(class_counts.index, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, count in zip(bars, class_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom')
        
        # 饼图
        wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_counts.index, 
                                          autopct='%1.1f%%', colors=self.colors[:len(class_counts)],
                                          startangle=90)
        ax2.set_title('类别比例分布')
        
        # 美化饼图文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # 保存图表
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 类别分布图已保存到: {save_path}")
    
    def create_sample_predictions(self, model_path, test_images_dir, num_samples=6, save_path="outputs/results/sample_predictions.png"):
        """创建样本预测展示"""
        print("🖼️ 创建样本预测展示...")
        
        try:
            # 加载模型
            model = YOLO(model_path)
            
            # 获取测试图片
            test_dir = Path(test_images_dir)
            if not test_dir.exists():
                print(f"❌ 测试图片目录不存在: {test_dir}")
                return
            
            image_files = list(test_dir.glob("*.jpg"))[:num_samples]
            if len(image_files) < num_samples:
                print(f"⚠️ 测试图片数量不足，只找到 {len(image_files)} 张图片")
            
            # 创建子图
            cols = 3
            rows = (len(image_files) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            fig.suptitle('YOLOv8 水稻病害检测样本预测结果', fontsize=16, fontweight='bold')
            
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, image_path in enumerate(image_files):
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                
                # 读取图片
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 进行预测
                results = model(image_path, conf=0.5)
                
                # 绘制检测结果
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # 绘制边界框
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                   linewidth=2, edgecolor=self.colors[class_id], 
                                                   facecolor='none')
                            ax.add_patch(rect)
                            
                            # 添加标签
                            label = f"{self.class_names[class_id]}: {confidence:.2f}"
                            ax.text(x1, y1-5, label, fontsize=8, color=self.colors[class_id],
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                ax.imshow(image)
                ax.set_title(f"样本 {i+1}: {image_path.name}", fontsize=10)
                ax.axis('off')
            
            # 隐藏多余的子图
            for i in range(len(image_files), rows * cols):
                row = i // cols
                col = i % cols
                if rows > 1:
                    axes[row, col].axis('off')
                else:
                    axes[col].axis('off')
            
            plt.tight_layout()
            
            # 保存图表
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"💾 样本预测展示已保存到: {save_path}")
            
        except Exception as e:
            print(f"❌ 创建样本预测时出错: {str(e)}")
    
    def generate_report(self, model_path="outputs/models/rice_disease_detection/weights/best.pt"):
        """生成完整的可视化报告"""
        print("📋 生成可视化报告...")
        
        # 创建输出目录
        output_dir = Path("outputs/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 绘制训练曲线
            self.plot_training_curves(output_dir / "training_curves.png")
            
            # 2. 绘制类别分布
            self.plot_class_distribution(save_path=str(output_dir / "class_distribution.png"))
            
            # 3. 绘制混淆矩阵
            self.plot_confusion_matrix(model_path, "data.yaml", 
                                     save_path=str(output_dir / "confusion_matrix.png"))
            
            # 4. 创建样本预测展示
            test_dir = "outputs/yolo_dataset/test/images"
            self.create_sample_predictions(model_path, test_dir, 
                                         save_path=str(output_dir / "sample_predictions.png"))
            
            print(f"\n✅ 所有可视化结果已保存到: {output_dir}")
            print("📁 生成的文件:")
            for file in output_dir.glob("*.png"):
                print(f"  - {file.name}")
                
        except Exception as e:
            print(f"❌ 生成报告时出错: {str(e)}")

def main():
    """主函数"""
    print("📊 YOLOv8 水稻病害检测结果可视化")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = ResultsVisualizer()
    
    # 检查模型是否存在
    model_path = "outputs/models/rice_disease_detection/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 train_yolov8.py 训练模型")
        return
    
    # 生成完整报告
    visualizer.generate_report(model_path)

if __name__ == "__main__":
    main()
