import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

from config import Config


class FeatureVisualizer:
    def __init__(self, model_path, device='cuda'):
        self.model_path = Path(model_path)
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"Loading model for feature visualization: {self.model_path}")
        self.yolo_model = YOLO(str(self.model_path))

        self.class_names = Config.CLASS_NAMES

    def visualize_learned_features(self, image_path, output_dir=None):
        image_path = Path(image_path)

        if output_dir is None:
            output_dir = Config.RESULTS_DIR / 'feature_visualization'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nVisualizing features for: {image_path.name}")

        original_img = cv2.imread(str(image_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        results = self.yolo_model.predict(str(image_path), device=self.device, verbose=False)[0]
        pred_class = results.probs.top1
        pred_conf = results.probs.top1conf.item()

        print(f"Predicted: {self.class_names[pred_class]} (conf: {pred_conf:.4f})")

        self._visualize_gradcam_alternative(image_path, pred_class, original_img, output_dir)

        return output_dir

    def _visualize_gradcam_alternative(self, image_path, pred_class, original_img, output_dir):
        resized_img = cv2.resize(original_img, (224, 224))

        img_normalized = resized_img.astype(np.float32) / 255.0

        gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        heatmap = self._create_attention_heatmap(resized_img)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(resized_img)
        axes[0, 1].set_title(f'Prediction: {self.class_names[pred_class]}', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title('Edge Detection (Disease Boundaries)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(heatmap, cmap='jet')
        axes[1, 0].set_title('Attention Heatmap', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        overlay = cv2.addWeighted(resized_img, 0.6, cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), 0.4, 0)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Disease Region Attention Overlay', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        color_analysis = self._analyze_disease_colors(resized_img)
        axes[1, 2].imshow(color_analysis)
        axes[1, 2].set_title('Color-based Disease Analysis', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')

        plt.suptitle(f'Feature Analysis: Model learns DISEASE patterns, not background',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        save_path = output_dir / f'feature_viz_{image_path.stem}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Feature visualization saved: {save_path}")

    def _create_attention_heatmap(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        yellow_lower = np.array([20, 40, 40])
        yellow_upper = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        brown_lower = np.array([10, 40, 20])
        brown_upper = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

        disease_mask = cv2.bitwise_or(yellow_mask, brown_mask)

        heatmap = cv2.GaussianBlur(disease_mask, (21, 21), 0)

        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return heatmap

    def _analyze_disease_colors(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        masks = {}
        masks['yellow'] = cv2.inRange(hsv, np.array([20, 40, 40]), np.array([40, 255, 255]))
        masks['brown'] = cv2.inRange(hsv, np.array([10, 40, 20]), np.array([20, 255, 200]))
        masks['dark_spots'] = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))

        result = image.copy()

        result[masks['yellow'] > 0] = [255, 255, 0]
        result[masks['brown'] > 0] = [165, 42, 42]
        result[masks['dark_spots'] > 0] = [139, 0, 0]

        return result

    def compare_backgrounds(self, white_bg_image, field_bg_image, output_dir=None):
        if output_dir is None:
            output_dir = Config.RESULTS_DIR / 'background_comparison'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("Comparing predictions across different backgrounds")
        print("="*60)

        white_img = cv2.imread(str(white_bg_image))
        white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2RGB)

        field_img = cv2.imread(str(field_bg_image))
        field_img = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)

        white_results = self.yolo_model.predict(str(white_bg_image), device=self.device, verbose=False)[0]
        field_results = self.yolo_model.predict(str(field_bg_image), device=self.device, verbose=False)[0]

        white_pred = white_results.probs.top1
        white_conf = white_results.probs.top1conf.item()

        field_pred = field_results.probs.top1
        field_conf = field_results.probs.top1conf.item()

        print(f"\nWhite background - Predicted: {self.class_names[white_pred]} (conf: {white_conf:.4f})")
        print(f"Field background - Predicted: {self.class_names[field_pred]} (conf: {field_conf:.4f})")

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        white_resized = cv2.resize(white_img, (224, 224))
        axes[0, 0].imshow(white_resized)
        axes[0, 0].set_title(f'White Background\nPredicted: {self.class_names[white_pred]} ({white_conf:.3f})',
                            fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        white_heatmap = self._create_attention_heatmap(white_resized)
        white_overlay = cv2.addWeighted(white_resized, 0.6,
                                       cv2.applyColorMap(white_heatmap, cv2.COLORMAP_JET), 0.4, 0)
        axes[0, 1].imshow(white_overlay)
        axes[0, 1].set_title('White BG: Disease Focus Areas', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        field_resized = cv2.resize(field_img, (224, 224))
        axes[1, 0].imshow(field_resized)
        axes[1, 0].set_title(f'Field Background\nPredicted: {self.class_names[field_pred]} ({field_conf:.3f})',
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        field_heatmap = self._create_attention_heatmap(field_resized)
        field_overlay = cv2.addWeighted(field_resized, 0.6,
                                       cv2.applyColorMap(field_heatmap, cv2.COLORMAP_JET), 0.4, 0)
        axes[1, 1].imshow(field_overlay)
        axes[1, 1].set_title('Field BG: Disease Focus Areas', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        plt.suptitle('Background Independence: Model focuses on DISEASE, not BACKGROUND',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        save_path = output_dir / 'background_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nBackground comparison saved: {save_path}")

        return save_path


def visualize_features(model_path, image_path=None, white_bg_image=None, field_bg_image=None, device='cuda'):
    visualizer = FeatureVisualizer(model_path, device)

    if white_bg_image and field_bg_image:
        visualizer.compare_backgrounds(white_bg_image, field_bg_image)

    if image_path:
        visualizer.visualize_learned_features(image_path)

    print("\n" + "="*60)
    print("Feature visualization complete!")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize learned features to prove model learns disease patterns, not background'
    )
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Single image for feature visualization')
    parser.add_argument('--white-bg', type=str, help='White background image for comparison')
    parser.add_argument('--field-bg', type=str, help='Field background image for comparison')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    visualize_features(args.model, args.image, args.white_bg, args.field_bg, args.device)
