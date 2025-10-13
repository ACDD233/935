#!/usr/bin/env python3
"""
YOLOv8 Rice Disease Detection Inference Script
水稻病害检测推理脚本
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

class RiceDiseaseDetector:
    """水稻病害检测器"""
    
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        初始化检测器
        
        Args:
            model_path: 训练好的模型路径
            confidence_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # 类别名称和颜色
        self.class_names = ['Brown Spot', 'Leaf Scald', 'Rice Blast', 'Rice Tungro', 'Sheath Blight']
        self.colors = [
            (255, 0, 0),    # Brown Spot - 红色
            (0, 255, 0),    # Leaf Scald - 绿色
            (0, 0, 255),    # Rice Blast - 蓝色
            (255, 255, 0),  # Rice Tungro - 黄色
            (255, 0, 255)   # Sheath Blight - 洋红色
        ]
    
    def detect_image(self, image_path, save_result=True, output_dir="outputs/results"):
        """
        检测单张图片
        
        Args:
            image_path: 图片路径
            save_result: 是否保存结果
            output_dir: 输出目录
            
        Returns:
            results: 检测结果
            annotated_image: 标注后的图片
        """
        print(f"🔍 检测图片: {image_path}")
        
        # 加载图片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图片: {image_path}")
        
        # 进行检测
        results = self.model(image, conf=self.confidence_threshold)
        
        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id]
                    })
        
        # 绘制检测结果
        annotated_image = self.draw_detections(image.copy(), detections)
        
        # 保存结果
        if save_result:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存标注图片
            image_name = Path(image_path).stem
            result_path = output_path / f"{image_name}_detected.jpg"
            cv2.imwrite(str(result_path), annotated_image)
            print(f"💾 结果已保存到: {result_path}")
            
            # 保存检测信息
            info_path = output_path / f"{image_name}_info.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"图片: {image_path}\n")
                f.write(f"检测到 {len(detections)} 个病害区域:\n\n")
                for i, det in enumerate(detections, 1):
                    f.write(f"{i}. {det['class_name']}\n")
                    f.write(f"   置信度: {det['confidence']:.3f}\n")
                    f.write(f"   位置: {det['bbox']}\n\n")
            print(f"📄 检测信息已保存到: {info_path}")
        
        return detections, annotated_image
    
    def detect_batch(self, image_dir, output_dir="outputs/results"):
        """
        批量检测图片
        
        Args:
            image_dir: 图片目录
            output_dir: 输出目录
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"图片目录不存在: {image_dir}")
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"⚠️ 在目录 {image_dir} 中未找到图片文件")
            return
        
        print(f"📁 找到 {len(image_files)} 张图片，开始批量检测...")
        
        all_results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {image_path.name}")
            try:
                detections, _ = self.detect_image(image_path, save_result=True, output_dir=output_dir)
                all_results.append({
                    'image_path': str(image_path),
                    'detections': detections
                })
            except Exception as e:
                print(f"❌ 处理图片 {image_path} 时出错: {str(e)}")
        
        # 保存汇总结果
        self.save_batch_summary(all_results, output_dir)
        print(f"\n✅ 批量检测完成! 结果保存在: {output_dir}")
    
    def draw_detections(self, image, detections):
        """
        在图片上绘制检测结果
        
        Args:
            image: 输入图片
            detections: 检测结果列表
            
        Returns:
            annotated_image: 标注后的图片
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # 获取颜色
            color = self.colors[class_id % len(self.colors)]
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 绘制标签文字
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def save_batch_summary(self, all_results, output_dir):
        """保存批量检测汇总结果"""
        summary_path = Path(output_dir) / "batch_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("水稻病害检测批量结果汇总\n")
            f.write("=" * 50 + "\n\n")
            
            total_detections = 0
            class_counts = {name: 0 for name in self.class_names}
            
            for result in all_results:
                image_path = Path(result['image_path']).name
                detections = result['detections']
                
                f.write(f"图片: {image_path}\n")
                f.write(f"检测到 {len(detections)} 个病害区域:\n")
                
                for det in detections:
                    class_name = det['class_name']
                    confidence = det['confidence']
                    f.write(f"  - {class_name} (置信度: {confidence:.3f})\n")
                    class_counts[class_name] += 1
                    total_detections += 1
                
                f.write("\n")
            
            f.write("汇总统计:\n")
            f.write("-" * 30 + "\n")
            f.write(f"总图片数: {len(all_results)}\n")
            f.write(f"总检测数: {total_detections}\n\n")
            f.write("各类别检测数量:\n")
            for class_name, count in class_counts.items():
                f.write(f"  {class_name}: {count}\n")
        
        print(f"📊 批量检测汇总已保存到: {summary_path}")

def main():
    """主函数 - 示例用法"""
    print("🌾 YOLOv8 水稻病害检测推理")
    print("=" * 40)
    
    # 模型路径 (训练完成后会有这个文件)
    model_path = "outputs/models/rice_disease_detection/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 train_yolov8.py 训练模型")
        return
    
    # 创建检测器
    detector = RiceDiseaseDetector(model_path, confidence_threshold=0.5)
    
    # 示例: 检测单张图片
    # image_path = "path/to/your/image.jpg"
    # detections, annotated_image = detector.detect_image(image_path)
    
    # 示例: 批量检测
    # image_dir = "path/to/image/directory"
    # detector.detect_batch(image_dir)
    
    print("💡 使用方法:")
    print("1. 检测单张图片:")
    print("   detector = RiceDiseaseDetector(model_path)")
    print("   detections, image = detector.detect_image('image.jpg')")
    print("\n2. 批量检测:")
    print("   detector.detect_batch('image_directory')")

if __name__ == "__main__":
    main()
