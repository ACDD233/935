import os
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
from datetime import datetime

from config import Config


class ModelEvaluator:
    def __init__(self, model_path, device=None):
        self.model_path = Path(model_path)
        self.device = device if device is not None else Config.DEVICE

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.results = {}

    def evaluate_on_split(self, split='test', save_plots=True, save_dir=None):
        print(f"\n{'='*60}")
        print(f"Evaluating on {split} set")
        print(f"{'='*60}")

        if save_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = Config.RESULTS_DIR / f'eval_{split}_{timestamp}'
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        split_dir = Config.PROCESSED_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        results = self.model.val(
            data=str(Config.get_data_yaml_path()),
            split=split,
            device=self.device,
            plots=save_plots,
            save_json=True,
            project=str(save_dir.parent),
            name=save_dir.name,
            exist_ok=True,
            workers=0  # Disable multiprocessing to avoid connection reset errors
        )

        # Calculate Top-2 accuracy manually
        top2_acc = self._calculate_top2_accuracy(split_dir)

        metrics = self._extract_metrics(results, top2_acc)

        if save_plots:
            self._plot_confusion_matrix(results, save_dir, split)
            self._save_detailed_report(metrics, save_dir, split)

        self.results[split] = metrics

        print(f"\n{split.upper()} Results:")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Top-1 Accuracy: {metrics['top1_acc']:.4f}")
        print(f"Top-2 Accuracy: {metrics['top2_acc']:.4f}")

        return metrics

    def evaluate_scenarios(self, save_dir=None):
        if save_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = Config.RESULTS_DIR / f'eval_scenarios_{timestamp}'
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        scenarios = {
            'white_background': self._get_images_by_background('white'),
            'field_background': self._get_images_by_background('field'),
            'mixed': list(Config.PROCESSED_DIR / 'test')
        }

        scenario_results = {}

        for scenario_name, image_paths in scenarios.items():
            print(f"\n{'='*60}")
            print(f"Evaluating scenario: {scenario_name}")
            print(f"{'='*60}")

            if not image_paths:
                print(f"No images found for scenario: {scenario_name}")
                continue

            predictions = []
            ground_truths = []

            for img_path in image_paths:
                if isinstance(img_path, Path):
                    img_path = str(img_path)

                result = self.model.predict(img_path, device=self.device, verbose=False)[0]

                pred_class = result.probs.top1
                predictions.append(pred_class)

                true_class = self._get_true_class_from_path(img_path)
                ground_truths.append(true_class)

            metrics = self._calculate_metrics(ground_truths, predictions, scenario_name)
            scenario_results[scenario_name] = metrics

            self._plot_scenario_confusion_matrix(
                ground_truths, predictions, scenario_name, save_dir
            )

            print(f"\n{scenario_name.upper()} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Total samples: {metrics['total_samples']}")

        self._save_scenario_comparison(scenario_results, save_dir)

        return scenario_results

    def _get_images_by_background(self, bg_type):
        test_dir = Config.PROCESSED_DIR / 'test'
        images = []

        bg_prefix = 'wb' if bg_type == 'white' else 'f'

        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        if bg_type == 'white' and 'wb' in img_path.stem.lower():
                            images.append(img_path)
                        elif bg_type == 'field' and 'wb' not in img_path.stem.lower():
                            images.append(img_path)

        return images

    def _get_true_class_from_path(self, img_path):
        img_path = Path(img_path)
        class_name = img_path.parent.name

        if class_name in Config.CLASS_NAMES:
            return Config.CLASS_NAMES.index(class_name)
        return -1

    def _calculate_top2_accuracy(self, split_dir):
        """Calculate Top-2 accuracy by checking if true label is in top 2 predictions"""
        correct_top2 = 0
        total = 0

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            if class_name not in Config.CLASS_NAMES:
                continue

            true_label = Config.CLASS_NAMES.index(class_name)

            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue

                # Predict
                results = self.model.predict(str(img_path), device=self.device, verbose=False)[0]

                # Get top 2 predictions
                top5_indices = results.probs.top5  # Get top5, we'll use first 2
                top2_indices = top5_indices[:2] if len(top5_indices) >= 2 else top5_indices

                # Check if true label is in top 2
                if true_label in top2_indices:
                    correct_top2 += 1

                total += 1

        return correct_top2 / total if total > 0 else 0.0

    def _extract_metrics(self, results, top2_acc):
        metrics = {
            'top1_acc': float(results.top1) if hasattr(results, 'top1') else 0.0,
            'top2_acc': float(top2_acc),
            'accuracy': float(results.top1) if hasattr(results, 'top1') else 0.0,
        }

        return metrics

    def _calculate_metrics(self, y_true, y_pred, scenario_name):
        accuracy = accuracy_score(y_true, y_pred)

        return {
            'accuracy': float(accuracy),
            'total_samples': len(y_true),
            'scenario': scenario_name
        }

    def _plot_confusion_matrix(self, results, save_dir, split_name):
        save_path = save_dir / f'confusion_matrix_{split_name}.png'
        print(f"Confusion matrix saved to: {save_path}")

    def _plot_scenario_confusion_matrix(self, y_true, y_pred, scenario_name, save_dir):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=Config.CLASS_NAMES,
                    yticklabels=Config.CLASS_NAMES)
        plt.title(f'Confusion Matrix - {scenario_name.replace("_", " ").title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        save_path = save_dir / f'confusion_matrix_{scenario_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved: {save_path}")

    def _save_detailed_report(self, metrics, save_dir, split_name):
        report_path = save_dir / f'metrics_{split_name}.json'
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Detailed metrics saved: {report_path}")

    def _save_scenario_comparison(self, scenario_results, save_dir):
        comparison_path = save_dir / 'scenario_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(scenario_results, f, indent=2)

        scenarios = list(scenario_results.keys())
        accuracies = [scenario_results[s]['accuracy'] for s in scenarios]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(scenarios, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Scenario', fontsize=12)
        plt.title('Model Performance Across Different Scenarios', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()

        plot_path = save_dir / 'scenario_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nScenario comparison saved: {plot_path}")


def evaluate_model(model_path, split='test', scenarios=False):
    evaluator = ModelEvaluator(model_path)

    if scenarios:
        return evaluator.evaluate_scenarios()
    else:
        return evaluator.evaluate_on_split(split)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate rice disease classification model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--scenarios', action='store_true',
                        help='Evaluate on different scenarios (white/field/mixed)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    Config.DEVICE = args.device if args.device else Config.DEVICE

    evaluate_model(args.model, args.split, args.scenarios)
