import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from datetime import datetime
import shutil

from config import Config


class RiceDiseaseTrainer:
    def __init__(self, model_name='yolov8s-cls.pt', imgsz=224, batch_size=16,
                 epochs=100, patience=20, device=None, save_dir=None):
        self.model_name = model_name
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device if device is not None else Config.DEVICE

        if save_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_dir = Config.RESULTS_DIR / f'train_{timestamp}'
        else:
            self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.results = None

    def prepare_data_yaml(self):
        # For YOLO classification, we need to create a proper directory structure
        # and use the directory path directly, not a YAML file
        data_yaml = {
            'path': str(Config.PROCESSED_DIR.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'names': {i: name for i, name in enumerate(Config.CLASS_NAMES)}
        }

        yaml_path = Config.PROCESSED_DIR / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"Data YAML created at: {yaml_path}")
        return str(Config.PROCESSED_DIR.absolute())  # Return directory path, not YAML path

    def train(self, resume=False):
        print(f"\n{'='*60}")
        print(f"Starting training with YOLOv8s-cls")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Image size: {self.imgsz}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Patience: {self.patience}")
        print(f"Save directory: {self.save_dir}")
        print(f"{'='*60}\n")

        data_path = self.prepare_data_yaml()

        self.model = YOLO(self.model_name)

        self.results = self.model.train(
            data=data_path,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=self.device,
            patience=self.patience,
            save=True,
            save_period=Config.SAVE_PERIOD,
            project=str(self.save_dir.parent),
            name=self.save_dir.name,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=Config.RANDOM_SEED,
            deterministic=True,
            val=True,
            plots=True,
            resume=resume
        )

        best_model_path = self.save_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            dest_path = Config.MODELS_DIR / 'best_model.pt'
            shutil.copy2(best_model_path, dest_path)
            print(f"\nBest model saved to: {dest_path}")

        print(f"\nTraining completed!")
        print(f"Results saved in: {self.save_dir}")

        return self.results

    def get_best_model_path(self):
        best_path = self.save_dir / 'weights' / 'best.pt'
        if best_path.exists():
            return best_path
        return None


def train_model(epochs=100, batch_size=16, imgsz=224, device=None, resume=False):
    Config.ensure_directories()

    trainer = RiceDiseaseTrainer(
        model_name=Config.YOLO_MODEL,
        imgsz=imgsz,
        batch_size=batch_size,
        epochs=epochs,
        patience=Config.PATIENCE,
        device=device
    )

    results = trainer.train(resume=resume)

    return trainer, results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLOv8s-cls for rice disease classification')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=224, help='Input image size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume
    )
