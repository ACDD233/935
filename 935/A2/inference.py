import os
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

from config import Config


class DiseasePredictor:
    def __init__(self, model_path, device=None, conf_threshold=0.5):
        self.model_path = Path(model_path)
        self.device = device if device is not None else 'cpu'
        self.conf_threshold = conf_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"Loading model from: {self.model_path}")
        print(f"Using device: {self.device}")

        self.model = YOLO(str(self.model_path))
        self.class_names = Config.CLASS_NAMES

    def predict_single(self, image_path, save_result=False, output_dir=None):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\nPredicting: {image_path.name}")

        results = self.model.predict(
            source=str(image_path),
            device=self.device,
            verbose=False
        )[0]

        prediction = self._parse_prediction(results)

        print(f"Predicted class: {prediction['class_name']}")
        print(f"Confidence: {prediction['confidence']:.4f}")

        if save_result:
            if output_dir is None:
                output_dir = Config.RESULTS_DIR / 'predictions'
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self._save_prediction_visualization(image_path, prediction, output_dir)

        return prediction

    def predict_folder(self, folder_path, save_results=True, output_dir=None):
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in folder_path.iterdir()
                      if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No images found in {folder_path}")
            return []

        print(f"\nFound {len(image_files)} images in {folder_path}")
        print(f"Starting batch prediction...\n")

        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Config.RESULTS_DIR / f'predictions_{timestamp}'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = []

        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")

            try:
                pred = self.predict_single(image_path, save_result=False)
                pred['image_path'] = str(image_path)
                predictions.append(pred)

            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue

        if save_results:
            self._save_batch_results(predictions, image_files, output_dir)

        self._print_summary(predictions)

        return predictions

    def _parse_prediction(self, results):
        probs = results.probs

        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()

        top5_indices = probs.top5
        top5_confs = probs.top5conf.tolist()

        # Extract only Top-2 predictions
        top2_indices = top5_indices[:2]
        top2_confs = top5_confs[:2]

        prediction = {
            'class_id': int(top1_idx),
            'class_name': self.class_names[top1_idx],
            'confidence': float(top1_conf),
            'top2_classes': [
                {
                    'class_id': int(idx),
                    'class_name': self.class_names[idx],
                    'confidence': float(conf)
                }
                for idx, conf in zip(top2_indices, top2_confs)
            ]
        }

        return prediction

    def _save_prediction_visualization(self, image_path, prediction, output_dir):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f'Input Image: {image_path.name}', fontsize=12)

        classes = [item['class_name'] for item in prediction['top2_classes']]
        confidences = [item['confidence'] for item in prediction['top2_classes']]

        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(classes))]

        y_pos = np.arange(len(classes))
        ax2.barh(y_pos, confidences, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes)
        ax2.invert_yaxis()
        ax2.set_xlabel('Confidence', fontsize=11)
        ax2.set_title('Top-2 Predictions', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 1.0)

        for i, v in enumerate(confidences):
            ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

        plt.tight_layout()

        output_path = output_dir / f'pred_{image_path.stem}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved: {output_path}")

    def _save_batch_results(self, predictions, image_files, output_dir):
        results_file = output_dir / 'predictions.txt'

        with open(results_file, 'w') as f:
            f.write(f"Rice Disease Classification Results\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total images: {len(image_files)}\n")
            f.write(f"Successful predictions: {len(predictions)}\n")
            f.write(f"{'='*80}\n\n")

            for pred in predictions:
                f.write(f"Image: {Path(pred['image_path']).name}\n")
                f.write(f"Predicted: {pred['class_name']} (confidence: {pred['confidence']:.4f})\n")
                f.write(f"\n")

        print(f"\nResults saved to: {results_file}")

        self._plot_class_distribution(predictions, output_dir)

    def _plot_class_distribution(self, predictions, output_dir):
        class_counts = {name: 0 for name in self.class_names}

        for pred in predictions:
            class_counts[pred['class_name']] += 1

        plt.figure(figsize=(12, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        bars = plt.bar(classes, counts, color='#3498db', edgecolor='black', linewidth=1.2)
        plt.xlabel('Disease Class', fontsize=12)
        plt.ylabel('Number of Predictions', fontsize=12)
        plt.title('Distribution of Predicted Classes', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        plot_path = output_dir / 'class_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Class distribution plot saved: {plot_path}")

    def _print_summary(self, predictions):
        if not predictions:
            print("\nNo successful predictions.")
            return

        print(f"\n{'='*60}")
        print(f"Prediction Summary")
        print(f"{'='*60}")

        class_counts = {}
        for pred in predictions:
            class_name = pred['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = (count / len(predictions)) * 100
            print(f"{class_name}: {count} ({percentage:.1f}%)")

        avg_confidence = np.mean([p['confidence'] for p in predictions])
        print(f"\nAverage confidence: {avg_confidence:.4f}")
        print(f"{'='*60}")


def predict(model_path, source, device='cpu', save_results=True):
    predictor = DiseasePredictor(model_path, device=device)

    source_path = Path(source)

    if source_path.is_file():
        return predictor.predict_single(source_path, save_result=save_results)
    elif source_path.is_dir():
        return predictor.predict_folder(source_path, save_results=save_results)
    else:
        raise ValueError(f"Invalid source: {source}. Must be a file or directory.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict rice disease from images')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image file or folder')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save prediction results')

    args = parser.parse_args()

    predict(args.model, args.source, args.device, args.save)
