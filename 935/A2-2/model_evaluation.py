#!/usr/bin/env python3
"""
Model evaluation - comprehensive analysis and reporting
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import torch
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Model evaluator."""
    
    def __init__(self, model_path, data_yaml_path="data.yaml"):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model weights.
            data_yaml_path: Path to dataset YAML.
        """
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.model = None
        self.class_names = ['Brown Spot', 'Leaf Scald', 'Rice Blast', 'Rice Tungro', 'Sheath Blight']
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load model from disk."""
        print("Loading model...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        print(f"Model loaded: {self.model_path}")
    
    def evaluate_on_dataset(self, split='val'):
        """Evaluate model on specified split."""
        print(f"Evaluating on split: {split}...")
        
        # Run validation
        results = self.model.val(data=self.data_yaml_path, split=split)
        
        # Extract metrics
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6)
        }
        
        print(f"Metrics on {split}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics, results
    
    def analyze_class_performance(self, results):
        """Analyze per-class metrics."""
        print("Analyzing per-class performance...")
        
        # AP values per class
        if hasattr(results.box, 'ap50'):
            # Handle both tensor and numpy array cases
            ap50_data = results.box.ap50
            ap_data = results.box.ap
            
            # Convert to numpy if needed
            if hasattr(ap50_data, 'cpu'):
                ap50_values = ap50_data.cpu().numpy()
            else:
                ap50_values = np.array(ap50_data)
                
            if hasattr(ap_data, 'cpu'):
                ap50_95_values = ap_data.cpu().numpy()
            else:
                ap50_95_values = np.array(ap_data)
            
            class_performance = pd.DataFrame({
                'Class': self.class_names,
                'AP@0.5': ap50_values,
                'AP@0.5:0.95': ap50_95_values
            })
            
            print("\nPer-class performance:")
            print(class_performance.to_string(index=False))
            
            return class_performance
        else:
            print("Warning: detailed per-class metrics not available")
            return None
    
    def plot_performance_metrics(self, metrics, save_path="outputs/results/performance_metrics.png"):
        """Plot overall metrics to an image."""
        print("Plotting performance metrics...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Overview', fontsize=16, fontweight='bold')
        
        # 1) Main metrics - bar chart
        main_metrics = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1_score']
        values = [metrics[metric] for metric in main_metrics]
        
        bars = axes[0, 0].bar(main_metrics, values, color=self.colors[:len(main_metrics)])
        axes[0, 0].set_title('Main Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 2) Radar chart
        angles = np.linspace(0, 2 * np.pi, len(main_metrics), endpoint=False).tolist()
        values_radar = values + values[:1]  # 闭合图形
        angles += angles[:1]
        
        axes[0, 1].plot(angles, values_radar, 'o-', linewidth=2, color='#FF6B6B')
        axes[0, 1].fill(angles, values_radar, alpha=0.25, color='#FF6B6B')
        axes[0, 1].set_xticks(angles[:-1])
        axes[0, 1].set_xticklabels(main_metrics)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title('Radar')
        axes[0, 1].grid(True)
        
        # 3) Horizontal comparison
        axes[1, 0].barh(main_metrics, values, color=self.colors[:len(main_metrics)])
        axes[1, 0].set_title('Comparison')
        axes[1, 0].set_xlabel('Score')
        
        # 4) Performance level pie
        performance_levels = []
        for value in values:
            if value >= 0.9:
                performance_levels.append('Excellent')
            elif value >= 0.8:
                performance_levels.append('Good')
            elif value >= 0.7:
                performance_levels.append('Fair')
            else:
                performance_levels.append('Needs improvement')
        
        level_counts = pd.Series(performance_levels).value_counts()
        axes[1, 1].pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
                      colors=self.colors[:len(level_counts)])
        axes[1, 1].set_title('Performance Levels')
        
        plt.tight_layout()
        
        # Save figure
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved metrics plot: {save_path}")
    
    def plot_class_performance(self, class_performance, save_path="outputs/results/class_performance.png"):
        """Plot per-class performance."""
        if class_performance is None:
            return
        
        print("Plotting per-class charts...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Per-class Performance', fontsize=16, fontweight='bold')
        
        # AP@0.5 柱状图
        bars1 = axes[0].bar(class_performance['Class'], class_performance['AP@0.5'], 
                           color=self.colors[:len(class_performance)])
        axes[0].set_title('AP@0.5 by Class')
        axes[0].set_ylabel('AP@0.5')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Labels
        for bar, value in zip(bars1, class_performance['AP@0.5']):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # AP@0.5:0.95 柱状图
        bars2 = axes[1].bar(class_performance['Class'], class_performance['AP@0.5:0.95'], 
                           color=self.colors[:len(class_performance)])
        axes[1].set_title('AP@0.5:0.95 by Class')
        axes[1].set_ylabel('AP@0.5:0.95')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Labels
        for bar, value in zip(bars2, class_performance['AP@0.5:0.95']):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved class performance plot: {save_path}")
    
    def generate_detailed_report(self, metrics, class_performance, save_path="outputs/results/evaluation_report.json"):
        """Generate detailed evaluation report."""
        print("Generating evaluation report...")
        
        report = {
            'model_info': {
                'model_path': str(self.model_path),
                'data_yaml': str(self.data_yaml_path),
                'class_names': self.class_names
            },
            'overall_metrics': metrics,
            'class_performance': class_performance.to_dict('records') if class_performance is not None else None,
            'evaluation_summary': {
                'best_performing_class': None,
                'worst_performing_class': None,
                'overall_grade': None
            }
        }
        
        # Best and worst classes
        if class_performance is not None:
            best_idx = class_performance['AP@0.5'].idxmax()
            worst_idx = class_performance['AP@0.5'].idxmin()
            
            report['evaluation_summary']['best_performing_class'] = {
                'class': class_performance.loc[best_idx, 'Class'],
                'ap50': class_performance.loc[best_idx, 'AP@0.5']
            }
            report['evaluation_summary']['worst_performing_class'] = {
                'class': class_performance.loc[worst_idx, 'Class'],
                'ap50': class_performance.loc[worst_idx, 'AP@0.5']
            }
        
        # Grade by mAP50
        avg_ap50 = metrics['mAP50']
        if avg_ap50 >= 0.9:
            grade = 'A+'
        elif avg_ap50 >= 0.8:
            grade = 'A'
        elif avg_ap50 >= 0.7:
            grade = 'B'
        elif avg_ap50 >= 0.6:
            grade = 'C'
        else:
            grade = 'D'
        
        report['evaluation_summary']['overall_grade'] = grade
        
        # Save report
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Saved evaluation report: {save_path}")
        
        # Summary
        print("\nSummary:")
        print(f"  Overall grade: {grade}")
        print(f"  mAP@0.5: {avg_ap50:.4f}")
        if class_performance is not None:
            print(f"  Best class: {report['evaluation_summary']['best_performing_class']['class']}")
            print(f"  Worst class: {report['evaluation_summary']['worst_performing_class']['class']}")
        
        return report
    
    def run_complete_evaluation(self):
        """Run full evaluation pipeline."""
        print("Starting full evaluation...")
        print("=" * 50)
        
        try:
            # 1) Evaluate on val
            metrics, results = self.evaluate_on_dataset('val')
            
            # 2) Per-class analysis
            class_performance = self.analyze_class_performance(results)
            
            # 3) Plot overall metrics
            self.plot_performance_metrics(metrics)
            
            # 4) Plot per-class metrics
            self.plot_class_performance(class_performance)
            
            # 5) Generate report
            report = self.generate_detailed_report(metrics, class_performance)
            
            print("\nEvaluation finished.")
            print("Results saved under: outputs/results/")
            
            return report
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise

def main():
    """CLI entrypoint."""
    print("YOLOv8 Model Evaluation")
    print("=" * 40)
    
    model_path = "outputs/models/rice_disease_detection/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Train a model first.")
        return
    
    evaluator = ModelEvaluator(model_path)
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
