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
from yolo_gradcam import YOLOGradCAM


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

    def _find_target_layer_for_api(self, torch_model):
        """为 pytorch-grad-cam API 找到合适的目标层"""
        for name, module in torch_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # 选择较深的卷积层
                if any(keyword in name.lower() for keyword in ['backbone', 'neck', 'head']):
                    return module
        return None

    def _preprocess_for_api(self, image_path):
        """为 API 预处理图像"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 保存原始尺寸
        original_size = img.shape[:2]
        
        # 调整尺寸
        img_resized = cv2.resize(img, (self.config.imgsz, self.config.imgsz))
        
        # 归一化
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 转换为张量
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        return img_tensor, img_resized, original_size

    def generate_gradcam(self, model, image_path, target_class=None):
        """Generate Grad-CAM heatmap using pytorch-grad-cam API and custom YOLO implementation"""
        # 方法1: 尝试使用 pytorch-grad-cam API (推荐)
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            
            # 获取 PyTorch 模型
            torch_model = model.model
            torch_model.eval()
            
            # 找到目标层
            target_layer = self._find_target_layer_for_api(torch_model)
            if target_layer is None:
                raise ValueError("No suitable target layer found")
            
            # 创建 Grad-CAM
            cam = GradCAM(model=torch_model, target_layers=[target_layer])
            
            # 预处理图像
            img_tensor, img_resized, original_size = self._preprocess_for_api(image_path)
            
            # 生成 CAM
            grayscale_cam = cam(input_tensor=img_tensor, targets=target_class)
            
            # 调整到原始尺寸
            gradcam_resized = cv2.resize(grayscale_cam[0], (original_size[1], original_size[0]))
            
            # 获取预测结果
            with torch.no_grad():
                outputs = torch_model(img_tensor)
                if hasattr(outputs, 'probs'):
                    pred_probs = outputs.probs.data.cpu().numpy()
                    pred_class = outputs.probs.top1
                else:
                    pred_probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]
                    pred_class = outputs.argmax(dim=1).item()
            
            print(f"  Successfully generated Grad-CAM using pytorch-grad-cam API for {os.path.basename(image_path)}")
            return gradcam_resized, pred_probs, pred_class
            
        except Exception as e:
            print(f"  pytorch-grad-cam API failed for {os.path.basename(image_path)}: {e}")
        
        # 方法2: 使用自定义的 YOLO Grad-CAM
        try:
            yolo_gradcam = YOLOGradCAM(model.ckpt_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            gradcam_map, pred_probs, pred_class = yolo_gradcam.generate_gradcam(image_path, target_class)
            
            if gradcam_map is not None:
                print(f"  Successfully generated custom YOLO attention for {os.path.basename(image_path)}")
                return gradcam_map, pred_probs, pred_class
            else:
                print(f"  Custom YOLO attention failed for {os.path.basename(image_path)}, trying fallback")
                return self.generate_gradcam_fallback(model, image_path, target_class)
                
        except Exception as e:
            print(f"  Custom YOLO attention error for {os.path.basename(image_path)}: {e}")
            return self.generate_gradcam_fallback(model, image_path, target_class)
    
    def generate_gradcam_fallback(self, model, image_path, target_class=None):
        """Fallback Grad-CAM method (original implementation)"""
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

        # Check if prediction is valid (not None)
        if prediction is not None:
            pred_class = prediction.argmax()
            pred_conf = prediction[pred_class]
            title = f'Prediction: {class_names[pred_class]}\nConfidence: {pred_conf:.3f}'
        else:
            # Fallback when prediction is None
            pred_class = 0
            pred_conf = 0.0
            title = 'Prediction: Failed\nConfidence: N/A'
        
        axes[2].set_title(title)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_fold(self, fold_num):
        """Generate visualizations for a single fold using true YOLO attention"""
        print(f"\nVisualizing Fold {fold_num} with true YOLO attention mechanism...")

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
                    else:
                        print(f"  Skipped {img_file}: Failed to generate heatmap")

                except Exception as e:
                    print(f"  Failed {img_file}: {e}")

        print(f"Fold {fold_num} visualization complete")

    def visualize_all_folds(self):
        """Generate visualizations for all folds using true YOLO attention"""
        print(f"\nStarting true YOLO attention visualization for {self.config.n_splits} folds")
        print("Each fold will use the actual YOLO model's attention mechanism")

        for fold_num in range(1, self.config.n_splits + 1):
            self.visualize_fold(fold_num)

        print(f"\nTrue YOLO attention visualization complete. Saved to {self.visualizations_dir}\n")


def visualize_gradcam(config, fold_num=None):
    """Main function for YOLO attention visualization"""
    print("\n" + "="*60)
    print("YOLO ATTENTION MECHANISM VISUALIZATION")
    print("="*60)
    print("This visualization uses two methods to generate Grad-CAM heatmaps:")
    print("1. pytorch-grad-cam API (recommended)")
    print("2. Custom YOLO attention mechanism")
    print("3. Fallback method (original implementation)")
    print("="*60)
    
    visualizer = GradCAMVisualizer(config)

    if fold_num is not None:
        visualizer.visualize_fold(fold_num)
    else:
        visualizer.visualize_all_folds()


if __name__ == '__main__':
    from config import parse_args

    config = parse_args()
    visualize_gradcam(config)
