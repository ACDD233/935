"""
Training module for K-Fold cross-validation
"""
import os
import json
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


class Trainer:
    """YOLO model trainer"""

    def __init__(self, config):
        self.config = config
        self.dataset_dir = config.dataset_dir
        self.models_dir = config.models_dir
        self.results_dir = config.results_dir
        self.n_splits = config.n_splits

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.fold_results = []

    def train_single_fold(self, fold_num):
        """Train a single fold"""
        print(f"\nTraining Fold {fold_num}/{self.n_splits}")

        fold_data_dir = os.path.join(self.dataset_dir, f'fold_{fold_num}')
        if not os.path.exists(fold_data_dir):
            raise FileNotFoundError(f"Fold {fold_num} dataset not found: {fold_data_dir}")

        model = YOLO(self.config.model_name)

        train_args = {
            'data': fold_data_dir,
            'epochs': self.config.epochs,
            'imgsz': self.config.imgsz,
            'batch': self.config.batch_size,
            'device': self.config.device,
            'freeze': self.config.freeze,
            'patience': self.config.patience,
            'optimizer': self.config.optimizer,
            'lr0': self.config.lr0,
            'weight_decay': self.config.weight_decay,
            'momentum': self.config.momentum,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_momentum': self.config.warmup_momentum,
            'warmup_bias_lr': self.config.warmup_bias_lr,
            'augment': self.config.augment,
            'cache': self.config.cache,
            'workers': self.config.workers,
            'verbose': self.config.verbose,
            'plots': self.config.plots,
            'save': self.config.save,
            'save_period': self.config.save_period,
            'project': self.results_dir,
            'name': f'fold_{fold_num}',
            'exist_ok': True
        }

        if self.config.augment:
            train_args.update({
                'degrees': self.config.degrees,
                'translate': self.config.translate,
                'scale': self.config.scale,
                'fliplr': self.config.fliplr,
                'flipud': self.config.flipud,
                'mosaic': self.config.mosaic,
                'mixup': self.config.mixup,
            })

        try:
            results = model.train(**train_args)

            model_save_path = os.path.join(self.models_dir, f'fold_{fold_num}_best.pt')
            best_model_path = os.path.join(self.results_dir, f'fold_{fold_num}', 'weights', 'best.pt')
            
            if os.path.exists(best_model_path):
                import shutil
                shutil.copy2(best_model_path, model_save_path)

            last_model_path = os.path.join(self.results_dir, f'fold_{fold_num}', 'weights', 'last.pt')
            last_save_path = os.path.join(self.models_dir, f'fold_{fold_num}_last.pt')
            if os.path.exists(last_model_path):
                import shutil
                shutil.copy2(last_model_path, last_save_path)

            fold_result = {
                'fold': fold_num,
                'best_top1_accuracy': float(results.top1) if hasattr(results, 'top1') else None,
                'best_top5_accuracy': float(results.top5) if hasattr(results, 'top5') else None,
                'model_path': model_save_path,
            }

            self.fold_results.append(fold_result)

            if fold_result['best_top1_accuracy']:
                print(f"Fold {fold_num} complete - Top1: {fold_result['best_top1_accuracy']:.4f}")

            return fold_result

        except Exception as e:
            print(f"Error in Fold {fold_num}: {e}")
            return None

    def train_all_folds(self):
        """Train all folds"""
        print(f"\nStarting {self.n_splits}-Fold training (seed={self.config.random_seed})")

        for fold_num in range(1, self.n_splits + 1):
            result = self.train_single_fold(fold_num)
            if result is None:
                print(f"Skipping Fold {fold_num}")

        self.save_results()
        return self.fold_results

    def save_results(self):
        """Save training results"""
        if not self.fold_results:
            return

        top1_accuracies = [r['best_top1_accuracy'] for r in self.fold_results if r['best_top1_accuracy'] is not None]

        import numpy as np
        summary = {
            'random_seed': self.config.random_seed,
            'n_splits': self.n_splits,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config.to_dict(),
            'fold_results': self.fold_results,
        }

        if top1_accuracies:
            summary['top1_mean'] = float(np.mean(top1_accuracies))
            summary['top1_std'] = float(np.std(top1_accuracies))
            summary['top1_min'] = float(np.min(top1_accuracies))
            summary['top1_max'] = float(np.max(top1_accuracies))

        results_json_path = os.path.join(self.results_dir, 'training_summary.json')
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        df_results = pd.DataFrame(self.fold_results)
        results_csv_path = os.path.join(self.results_dir, 'training_results.csv')
        df_results.to_csv(results_csv_path, index=False, encoding='utf-8-sig')

        if top1_accuracies:
            print(f"\nTraining complete - Mean Top1: {summary['top1_mean']:.4f} ± {summary['top1_std']:.4f}\n")


def train_model(config, fold_num=None):
    """Main training function"""
    trainer = Trainer(config)

    if fold_num is not None:
        result = trainer.train_single_fold(fold_num)
        trainer.fold_results = [result] if result else []
        trainer.save_results()
        return result
    else:
        return trainer.train_all_folds()


if __name__ == '__main__':
    from config import parse_args
    config = parse_args()
    train_model(config)
