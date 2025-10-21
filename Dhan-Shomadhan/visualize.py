"""
Grad-CAM visualization module for model attention regions
"""
import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random


class GradCAMVisualizer:
    """Grad-CAM visualizer for YOLO classification models"""

    def __init__(self, config):
        self.config = config
        self.models_dir = config.models_dir
        self.dataset_dir = config.dataset_dir
        self.visualizations_dir = config.visualizations_dir
        self.num_samples = config.vis_num_samples

        os.makedirs(self.visualizations_dir, exist_ok=True)

    def get_last_conv_layer(self, model):
        """Get the last convolutional layer"""
        last_conv = None
        for name, module in model.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.modules.conv.Conv2d)):
                last_conv = (name, module)

        if last_conv is None:
            raise ValueError("No conv layer found")

        return last_conv

    def generate_gradcam(self, model, image_path, target_class=None):
        """Generate Grad-CAM heatmap for a single image"""
        img = Image.open(image_path).convert('RGB')
        original_img = np.array(img)

        img_tensor = self.preprocess_image(img, self.config.imgsz)
        torch_model = model.model

        features = []
        gradients = []

        def forward_hook(module, input, output):
            features.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        target_layer_name, target_layer = self.get_last_conv_layer(model)

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        torch_model.eval()
        img_tensor = img_tensor.to(next(torch_model.parameters()).device)
        output = torch_model(img_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        torch_model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        forward_handle.remove()
        backward_handle.remove()

        if len(gradients) > 0 and len(features) > 0:
            gradient = gradients[0].cpu().data.numpy()[0]
            feature = features[0].cpu().data.numpy()[0]

            weights = np.mean(gradient, axis=(1, 2))

            cam = np.zeros(feature.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * feature[i]

            cam = np.maximum(cam, 0)

            if cam.max() > 0:
                cam = cam / cam.max()

            cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))

            return cam, output.softmax(dim=1)[0].cpu().detach().numpy(), target_class
        else:
            print("  Warning: Failed to get features or gradients")
            return None, None, target_class

    def preprocess_image(self, img, size):
        """Preprocess image for YOLO model"""
        img = img.resize((size, size), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return img_tensor

    def visualize_cam(self, image_path, heatmap, prediction, class_names, save_path):
        """Visualize and save Grad-CAM results"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        axes[2].imshow(superimposed)

        pred_class = prediction.argmax()
        pred_conf = prediction[pred_class]
        title = f'Prediction: {class_names[pred_class]}\nConfidence: {pred_conf:.3f}'
        axes[2].set_title(title)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_fold(self, fold_num):
        """Generate visualizations for a single fold"""
        print(f"\nVisualizing Fold {fold_num}...")

        model_path = os.path.join(self.models_dir, f'fold_{fold_num}_best.pt')
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return

        model = YOLO(model_path)

        val_dir = os.path.join(self.dataset_dir, f'fold_{fold_num}', 'val')
        if not os.path.exists(val_dir):
            print(f"Val data not found: {val_dir}")
            return

        class_names = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])

        fold_vis_dir = os.path.join(self.visualizations_dir, f'fold_{fold_num}')
        os.makedirs(fold_vis_dir, exist_ok=True)

        samples_per_class = max(1, self.num_samples // len(class_names))

        for class_name in class_names:
            class_dir = os.path.join(val_dir, class_name)
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            selected_samples = random.sample(image_files, min(samples_per_class, len(image_files)))

            print(f"  Processing {class_name}: {len(selected_samples)} samples")

            for img_file in selected_samples:
                img_path = os.path.join(class_dir, img_file)

                try:
                    heatmap, prediction, pred_class = self.generate_gradcam(model, img_path)

                    if heatmap is not None:
                        save_name = f"{class_name}_{os.path.splitext(img_file)[0]}_gradcam.png"
                        save_path = os.path.join(fold_vis_dir, save_name)
                        self.visualize_cam(img_path, heatmap, prediction, class_names, save_path)

                except Exception as e:
                    print(f"  Failed {img_file}: {e}")

        print(f"Fold {fold_num} visualization complete")

    def visualize_all_folds(self):
        """Generate visualizations for all folds"""
        print(f"\nStarting Grad-CAM visualization for {self.config.n_splits} folds")

        for fold_num in range(1, self.config.n_splits + 1):
            self.visualize_fold(fold_num)

        print(f"\nVisualization complete. Saved to {self.visualizations_dir}\n")


def visualize_gradcam(config, fold_num=None):
    """Main function for Grad-CAM visualization"""
    visualizer = GradCAMVisualizer(config)

    if fold_num is not None:
        visualizer.visualize_fold(fold_num)
    else:
        visualizer.visualize_all_folds()


if __name__ == '__main__':
    from config import parse_args

    config = parse_args()
    visualize_gradcam(config)
