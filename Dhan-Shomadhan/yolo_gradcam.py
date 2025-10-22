"""
真正的 YOLO 注意力机制可视化
基于 YOLO 模型内部结构的 Grad-CAM 实现
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


class YOLOGradCAM:
    """真正的 YOLO Grad-CAM 实现"""
    
    def __init__(self, model_path, device='cuda'):
        self.model_path = Path(model_path)
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading YOLO model: {self.model_path}")
        self.yolo_model = YOLO(str(self.model_path))
        
        # 获取 PyTorch 模型
        self.torch_model = self.yolo_model.model
        self.torch_model.eval()
        
        # 存储 hook 信息
        self.features = {}
        self.gradients = {}
        self.hooks = []
        
    def register_hooks(self, target_layers):
        """注册多个目标层的 hook"""
        def forward_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook
        
        # 注册所有目标层
        for layer_name, layer in target_layers:
            forward_h = layer.register_forward_hook(forward_hook(layer_name))
            backward_h = layer.register_full_backward_hook(backward_hook(layer_name))
            self.hooks.extend([forward_h, backward_h])
    
    def remove_hooks(self):
        """移除所有 hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.features.clear()
        self.gradients.clear()
    
    def find_yolo_attention_layers(self):
        """找到 YOLO 模型中的注意力相关层"""
        attention_layers = []
        
        for name, module in self.torch_model.named_modules():
            # 查找卷积层（特征提取）
            if isinstance(module, torch.nn.Conv2d):
                # 优先选择较深的卷积层
                if any(keyword in name.lower() for keyword in ['backbone', 'neck', 'head']):
                    attention_layers.append((name, module))
            
            # 查找注意力模块
            elif hasattr(module, '__class__') and 'attention' in module.__class__.__name__.lower():
                attention_layers.append((name, module))
            
            # 查找 YOLO 特有的模块
            elif any(keyword in name.lower() for keyword in ['c2f', 'sppf', 'bottleneck']):
                attention_layers.append((name, module))
        
        # 按深度排序，选择最后几个
        attention_layers = sorted(attention_layers, key=lambda x: len(x[0].split('.')))
        return attention_layers[-3:] if len(attention_layers) >= 3 else attention_layers
    
    def preprocess_image(self, image_path, target_size=640):
        """预处理图像"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 保存原始尺寸
        original_size = img.shape[:2]
        
        # 调整尺寸
        img_resized = cv2.resize(img, (target_size, target_size))
        
        # 归一化
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 转换为张量
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor, img_resized, original_size
    
    def generate_gradcam(self, image_path, target_class=None):
        """生成真正的 YOLO Grad-CAM"""
        try:
            # 预处理图像
            img_tensor, img_resized, original_size = self.preprocess_image(image_path)
            
            # 找到注意力层
            attention_layers = self.find_yolo_attention_layers()
            if not attention_layers:
                raise ValueError("No attention layers found in YOLO model")
            
            print(f"Found {len(attention_layers)} attention layers:")
            for name, _ in attention_layers:
                print(f"  - {name}")
            
            # 注册 hook
            self.register_hooks(attention_layers)
            
            # 前向传播
            with torch.enable_grad():
                img_tensor.requires_grad_(True)
                outputs = self.torch_model(img_tensor)
                
                # 获取预测结果
                if hasattr(outputs, 'probs'):
                    # YOLO 分类模型
                    probs = outputs.probs
                    if target_class is None:
                        target_class = probs.top1
                    class_score = probs.top1conf
                else:
                    # 其他情况
                    if target_class is None:
                        target_class = outputs.argmax(dim=1).item()
                    class_score = outputs[0, target_class]
                
                # 反向传播
                self.torch_model.zero_grad()
                class_score.backward(retain_graph=True)
            
            # 生成 Grad-CAM
            gradcam_maps = []
            for layer_name, _ in attention_layers:
                if layer_name in self.features and layer_name in self.gradients:
                    gradcam_map = self.compute_gradcam(
                        self.features[layer_name], 
                        self.gradients[layer_name]
                    )
                    gradcam_maps.append((layer_name, gradcam_map))
            
            # 移除 hook
            self.remove_hooks()
            
            if not gradcam_maps:
                raise ValueError("Failed to generate any Grad-CAM maps")
            
            # 选择最佳的热图（通常是最后一个）
            best_layer_name, best_gradcam = gradcam_maps[-1]
            
            # 调整到原始图像尺寸
            gradcam_resized = cv2.resize(best_gradcam, (original_size[1], original_size[0]))
            
            # 获取预测概率
            if hasattr(outputs, 'probs'):
                pred_probs = outputs.probs.data.cpu().numpy()
            else:
                pred_probs = F.softmax(outputs, dim=1).cpu().detach().numpy()[0]
            
            print(f"Successfully generated Grad-CAM using layer: {best_layer_name}")
            return gradcam_resized, pred_probs, target_class
            
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            self.remove_hooks()
            return None, None, None
    
    def compute_gradcam(self, features, gradients):
        """计算 Grad-CAM 热图"""
        # 获取梯度的全局平均池化
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 加权特征图
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        
        # 应用 ReLU
        cam = F.relu(cam)
        
        # 归一化
        cam = cam.squeeze().cpu().detach().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def visualize_attention(self, image_path, output_path=None):
        """可视化注意力机制"""
        image_path = Path(image_path)
        
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_yolo_attention.png"
        
        # 生成 Grad-CAM
        gradcam_map, pred_probs, pred_class = self.generate_gradcam(image_path)
        
        if gradcam_map is None:
            print("Failed to generate attention map")
            return None
        
        # 加载原始图像
        original_img = cv2.imread(str(image_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始图像
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM 热图
        im1 = axes[1].imshow(gradcam_map, cmap='jet')
        axes[1].set_title('YOLO Attention Map (Grad-CAM)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 叠加图像
        gradcam_colored = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)
        gradcam_colored = cv2.cvtColor(gradcam_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_img, 0.6, gradcam_colored, 0.4, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Attention Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'True YOLO Attention Mechanism\nPredicted Class: {pred_class}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"YOLO attention visualization saved: {output_path}")
        return output_path


def visualize_yolo_attention(model_path, image_path, output_path=None, device='cuda'):
    """可视化 YOLO 模型的真正注意力机制"""
    visualizer = YOLOGradCAM(model_path, device)
    return visualizer.visualize_attention(image_path, output_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize true YOLO attention mechanism')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Output path for visualization')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    visualize_yolo_attention(args.model, args.image, args.output, args.device)
