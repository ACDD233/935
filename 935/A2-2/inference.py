#!/usr/bin/env python3
"""
YOLOv8 Rice Disease Detection - Inference
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
    """Rice disease object detector."""
    
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained model weights.
            confidence_threshold: Confidence threshold for predictions.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Class names and colors
        self.class_names = ['Brown Spot', 'Leaf Scald', 'Rice Blast', 'Rice Tungro', 'Sheath Blight']
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255)
        ]
    
    def detect_image(self, image_path, save_result=True, output_dir="outputs/results"):
        """
        Run detection for a single image.
        
        Args:
            image_path: Image path.
            save_result: Whether to save annotated result.
            output_dir: Output directory.
            
        Returns:
            detections: List of detection dicts.
            annotated_image: Image with drawn detections.
        """
        print(f"Detecting: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Inference
        results = self.model(image, conf=self.confidence_threshold)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id]
                    })
        
        # Draw detections
        annotated_image = self.draw_detections(image.copy(), detections)
        
        # Save results
        if save_result:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save annotated image
            image_name = Path(image_path).stem
            result_path = output_path / f"{image_name}_detected.jpg"
            cv2.imwrite(str(result_path), annotated_image)
            print(f"Saved annotated image: {result_path}")
            
            # Save detection info
            info_path = output_path / f"{image_name}_info.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Detections: {len(detections)}\n\n")
                for i, det in enumerate(detections, 1):
                    f.write(f"{i}. {det['class_name']}\n")
                    f.write(f"   confidence: {det['confidence']:.3f}\n")
                    f.write(f"   bbox: {det['bbox']}\n\n")
            print(f"Saved detection info: {info_path}")
        
        return detections, annotated_image
    
    def detect_batch(self, image_dir, output_dir="outputs/results"):
        """
        Run detection on a directory of images.
        
        Args:
            image_dir: Directory containing images.
            output_dir: Output directory.
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in: {image_dir}")
            return
        
        print(f"Found {len(image_files)} images. Running batch inference...")
        
        all_results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            try:
                detections, _ = self.detect_image(image_path, save_result=True, output_dir=output_dir)
                all_results.append({
                    'image_path': str(image_path),
                    'detections': detections
                })
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # Save batch summary
        self.save_batch_summary(all_results, output_dir)
        print(f"\nBatch inference complete. Results saved to: {output_dir}")
    
    def draw_detections(self, image, detections):
        """
        Draw detections on image.
        
        Args:
            image: Input image (BGR).
            detections: Detection list produced by detect_image.
            
        Returns:
            Annotated image.
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # Color by class
            color = self.colors[class_id % len(self.colors)]
            
            # Box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Label bg
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def save_batch_summary(self, all_results, output_dir):
        """Write aggregated batch summary to a text file."""
        summary_path = Path(output_dir) / "batch_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Batch Inference Summary\n")
            f.write("=" * 50 + "\n\n")
            
            total_detections = 0
            class_counts = {name: 0 for name in self.class_names}
            class_conf_sums = {name: 0.0 for name in self.class_names}
            all_confidences = []
            images_with_detections = 0
            
            for result in all_results:
                image_path = Path(result['image_path']).name
                detections = result['detections']
                
                f.write(f"Image: {image_path}\n")
                f.write(f"Detections: {len(detections)}\n")
                
                for det in detections:
                    class_name = det['class_name']
                    confidence = det['confidence']
                    f.write(f"  - {class_name} (confidence: {confidence:.3f})\n")
                    class_counts[class_name] += 1
                    class_conf_sums[class_name] += float(confidence)
                    total_detections += 1
                    all_confidences.append(float(confidence))
                
                f.write("\n")
                if len(detections) > 0:
                    images_with_detections += 1
            
            f.write("Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total images: {len(all_results)}\n")
            f.write(f"Total detections: {total_detections}\n\n")
            f.write("Detections per class:\n")
            for class_name, count in class_counts.items():
                f.write(f"  {class_name}: {count}\n")
            
            # Additional stats
            f.write("\nAdditional statistics:\n")
            f.write("-" * 30 + "\n")
            total_images = len(all_results)
            images_without_detections = total_images - images_with_detections
            avg_detections_per_image = (total_detections / total_images) if total_images > 0 else 0.0
            f.write(f"Images with detections: {images_with_detections}\n")
            f.write(f"Images without detections: {images_without_detections}\n")
            f.write(f"Avg detections per image: {avg_detections_per_image:.3f}\n")
            
            if len(all_confidences) > 0:
                overall_avg = float(np.mean(all_confidences))
                overall_med = float(np.median(all_confidences))
                overall_min = float(np.min(all_confidences))
                overall_max = float(np.max(all_confidences))
                f.write(f"Overall avg confidence: {overall_avg:.4f}\n")
                f.write(f"Overall median confidence: {overall_med:.4f}\n")
                f.write(f"Min confidence: {overall_min:.4f}\n")
                f.write(f"Max confidence: {overall_max:.4f}\n")
                
                f.write("\nAvg confidence per class:\n")
                for class_name in self.class_names:
                    cnt = class_counts[class_name]
                    if cnt > 0:
                        avg_conf = class_conf_sums[class_name] / cnt
                        f.write(f"  {class_name}: {avg_conf:.4f}\n")
                    else:
                        f.write(f"  {class_name}: no detections\n")
        
        print(f"Saved batch summary: {summary_path}")

def main():
    """Example usage."""
    print("YOLOv8 Rice Disease Inference")
    print("=" * 40)
    
    model_path = "outputs/models/rice_disease_detection3/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Run train_yolov8.py to train a model first.")
        return
    
    detector = RiceDiseaseDetector(model_path, confidence_threshold=0.5)
    
    # Example: single image
    # detections, annotated_image = detector.detect_image('image.jpg')
    
    # Example: directory
    detector.detect_batch('image_directory')
    
    print("Usage:")
    print("1) Single image:")
    print("   detector = RiceDiseaseDetector(model_path)")
    print("   detections, image = detector.detect_image('image.jpg')")
    print("\n2) Batch:")
    print("   detector.detect_batch('image_directory')")

if __name__ == "__main__":
    main()
