import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import shutil

from config import Config
from trainer import RiceDiseaseTrainer
from evaluator import ModelEvaluator


class KFoldCrossValidator:
    def __init__(self, n_splits=5, seed=42):
        self.n_splits = n_splits
        self.seed = seed
        self.fold_results = []

        self.cv_dir = Config.RESULTS_DIR / f'cross_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.cv_dir.mkdir(parents=True, exist_ok=True)

    def prepare_folds(self):
        print(f"\n{'='*80}")
        print(f"Preparing {self.n_splits}-Fold Cross Validation")
        print(f"{'='*80}\n")

        all_images = self._collect_all_images()

        labels = [img['disease_id'] for img in all_images]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        self.folds = []
        for fold_idx, (train_val_indices, test_indices) in enumerate(skf.split(all_images, labels)):
            train_val_images = [all_images[i] for i in train_val_indices]
            test_images = [all_images[i] for i in test_indices]

            train_val_labels = [img['disease_id'] for img in train_val_images]

            n_train = int(len(train_val_images) * 0.85)
            train_images = train_val_images[:n_train]
            val_images = train_val_images[n_train:]

            fold_data = {
                'fold_idx': fold_idx,
                'train': train_images,
                'val': val_images,
                'test': test_images
            }

            self.folds.append(fold_data)

            print(f"Fold {fold_idx + 1}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

        return self.folds

    def run_cross_validation(self, epochs=100, batch_size=16, device=None):
        print(f"\n{'='*80}")
        print(f"Running {self.n_splits}-Fold Cross Validation")
        print(f"{'='*80}\n")

        if not hasattr(self, 'folds') or not self.folds:
            self.prepare_folds()

        for fold_idx, fold_data in enumerate(self.folds):
            print(f"\n{'='*80}")
            print(f"Training Fold {fold_idx + 1}/{self.n_splits}")
            print(f"{'='*80}\n")

            fold_result = self._train_fold(fold_idx, fold_data, epochs, batch_size, device)

            self.fold_results.append(fold_result)

            self._save_intermediate_results()

        statistics = self._compute_statistics()

        self._save_final_results(statistics)

        return self.fold_results, statistics

    def _collect_all_images(self):
        all_images = []
        disease_to_id = {name: idx for idx, name in enumerate(Config.CLASS_NAMES)}

        dataset_dir = Config.DATASET_DIR

        for bg_type in ['Field Background', 'White Background']:
            bg_dir = dataset_dir / bg_type
            if not bg_dir.exists():
                continue

            for disease_dir in bg_dir.iterdir():
                if not disease_dir.is_dir():
                    continue

                disease_name = self._normalize_disease_name(disease_dir.name)

                if disease_name not in disease_to_id:
                    continue

                disease_id = disease_to_id[disease_name]

                for img_file in disease_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        all_images.append({
                            'path': img_file,
                            'disease': disease_name,
                            'disease_id': disease_id,
                            'background': bg_type.replace(' ', '_').lower()
                        })

        print(f"Collected {len(all_images)} images from dataset")
        return all_images

    def _normalize_disease_name(self, name):
        mapping = {
            'Browon Spot': 'Brown_Spot',
            'Brown Spot': 'Brown_Spot',
            'Leaf Scaled': 'Leaf_Scald',
            'Rice Blast': 'Rice_Blast',
            'Rice Turgro': 'Rice_Tungro',
            'Rice Tungro': 'Rice_Tungro',
            'Sheath Blight': 'Sheath_Blight',
            'Shath Blight': 'Sheath_Blight'
        }
        return mapping.get(name, name)

    def _train_fold(self, fold_idx, fold_data, epochs, batch_size, device):
        fold_dir = self.cv_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = fold_dir / 'dataset'
        self._create_fold_dataset(fold_data, dataset_dir)

        original_processed_dir = Config.PROCESSED_DIR
        Config.PROCESSED_DIR = dataset_dir

        print(f"\n[Fold {fold_idx + 1}] Training model...")
        trainer = RiceDiseaseTrainer(
            model_name=Config.YOLO_MODEL,
            imgsz=Config.IMG_SIZE,
            batch_size=batch_size,
            epochs=epochs,
            patience=Config.PATIENCE,
            device=device,
            save_dir=fold_dir / 'training'
        )

        trainer.prepare_data_yaml()
        results = trainer.train()

        best_model_path = trainer.get_best_model_path()

        print(f"\n[Fold {fold_idx + 1}] Evaluating model...")
        evaluator = ModelEvaluator(best_model_path, device=device)

        test_metrics = evaluator.evaluate_on_split(
            split='test',
            save_plots=True,
            save_dir=fold_dir / 'evaluation'
        )

        scenario_metrics = self._evaluate_scenarios(evaluator, fold_data, fold_dir)

        Config.PROCESSED_DIR = original_processed_dir

        fold_result = {
            'fold_idx': fold_idx,
            'test_accuracy': test_metrics.get('accuracy', 0.0),
            'test_top1': test_metrics.get('top1_acc', 0.0),
            'test_top2': test_metrics.get('top2_acc', 0.0),
            'scenarios': scenario_metrics
        }

        print(f"\n[Fold {fold_idx + 1}] Results:")
        print(f"  Test accuracy: {fold_result['test_accuracy']:.4f}")
        print(f"  White background: {scenario_metrics['white_background']:.4f}")
        print(f"  Field background: {scenario_metrics['field_background']:.4f}")
        print(f"  Mixed: {scenario_metrics['mixed']:.4f}")

        return fold_result

    def _create_fold_dataset(self, fold_data, dataset_dir):
        for split_name in ['train', 'val', 'test']:
            images = fold_data[split_name]

            for img_info in images:
                disease = img_info['disease']
                dest_dir = dataset_dir / split_name / disease
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_path = dest_dir / img_info['path'].name
                if not dest_path.exists():
                    shutil.copy2(img_info['path'], dest_path)

    def _evaluate_scenarios(self, evaluator, fold_data, fold_dir):
        test_images = fold_data['test']

        white_bg_images = [img for img in test_images if 'white' in img['background']]
        field_bg_images = [img for img in test_images if 'field' in img['background']]

        white_acc = self._evaluate_image_list(evaluator, white_bg_images)
        field_acc = self._evaluate_image_list(evaluator, field_bg_images)
        mixed_acc = self._evaluate_image_list(evaluator, test_images)

        return {
            'white_background': white_acc,
            'field_background': field_acc,
            'mixed': mixed_acc
        }

    def _evaluate_image_list(self, evaluator, image_list):
        if not image_list:
            return 0.0

        correct = 0
        total = len(image_list)

        for img_info in image_list:
            result = evaluator.model.predict(str(img_info['path']), verbose=False)[0]
            pred_class = result.probs.top1
            true_class = img_info['disease_id']

            if pred_class == true_class:
                correct += 1

        return correct / total if total > 0 else 0.0

    def _save_intermediate_results(self):
        results_file = self.cv_dir / 'fold_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.fold_results, f, indent=2)

    def _compute_statistics(self):
        print(f"\n{'='*80}")
        print("Computing Cross-Validation Statistics")
        print(f"{'='*80}\n")

        test_accs = [r['test_accuracy'] for r in self.fold_results]
        white_accs = [r['scenarios']['white_background'] for r in self.fold_results]
        field_accs = [r['scenarios']['field_background'] for r in self.fold_results]
        mixed_accs = [r['scenarios']['mixed'] for r in self.fold_results]

        statistics = {
            'n_folds': self.n_splits,
            'test_set': {
                'mean': float(np.mean(test_accs)),
                'std': float(np.std(test_accs)),
                'min': float(np.min(test_accs)),
                'max': float(np.max(test_accs)),
                'folds': test_accs
            },
            'white_background': {
                'mean': float(np.mean(white_accs)),
                'std': float(np.std(white_accs)),
                'min': float(np.min(white_accs)),
                'max': float(np.max(white_accs)),
                'folds': white_accs
            },
            'field_background': {
                'mean': float(np.mean(field_accs)),
                'std': float(np.std(field_accs)),
                'min': float(np.min(field_accs)),
                'max': float(np.max(field_accs)),
                'folds': field_accs
            },
            'mixed': {
                'mean': float(np.mean(mixed_accs)),
                'std': float(np.std(mixed_accs)),
                'min': float(np.min(mixed_accs)),
                'max': float(np.max(mixed_accs)),
                'folds': mixed_accs
            }
        }

        print("Cross-Validation Results (Mean ± Std):")
        print(f"  Test set:         {statistics['test_set']['mean']:.4f} ± {statistics['test_set']['std']:.4f}")
        print(f"  White background: {statistics['white_background']['mean']:.4f} ± {statistics['white_background']['std']:.4f}")
        print(f"  Field background: {statistics['field_background']['mean']:.4f} ± {statistics['field_background']['std']:.4f}")
        print(f"  Mixed:            {statistics['mixed']['mean']:.4f} ± {statistics['mixed']['std']:.4f}")

        return statistics

    def _save_final_results(self, statistics):
        stats_file = self.cv_dir / 'cv_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)

        print(f"\nStatistics saved to: {stats_file}")

        self._plot_cv_results(statistics)

        self._save_latex_table(statistics)

    def _plot_cv_results(self, statistics):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        scenarios = ['test_set', 'white_background', 'field_background', 'mixed']
        titles = ['Test Set', 'White Background', 'Field Background', 'Mixed']

        for idx, (scenario, title) in enumerate(zip(scenarios, titles)):
            ax = axes[idx // 2, idx % 2]

            fold_results = statistics[scenario]['folds']
            mean = statistics[scenario]['mean']
            std = statistics[scenario]['std']

            x = np.arange(1, self.n_splits + 1)
            ax.bar(x, fold_results, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.axhline(y=mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='red', label=f'±1 Std: {std:.4f}')

            ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_xticks(x)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            for i, v in enumerate(fold_results):
                ax.text(i + 1, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'{self.n_splits}-Fold Cross-Validation Results', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        plot_path = self.cv_dir / 'cv_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Cross-validation plot saved: {plot_path}")

        fig, ax = plt.subplots(figsize=(12, 7))

        scenario_names = ['Test Set', 'White BG', 'Field BG', 'Mixed']
        means = [statistics[s]['mean'] for s in scenarios]
        stds = [statistics[s]['std'] for s in scenarios]

        x_pos = np.arange(len(scenario_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color='#2ecc71',
                     edgecolor='black', linewidth=1.5, alpha=0.8)

        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Evaluation Scenario', fontsize=14, fontweight='bold')
        ax.set_title(f'{self.n_splits}-Fold Cross-Validation: Overall Performance\n(Mean ± Standard Deviation)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenario_names, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()

        summary_plot_path = self.cv_dir / 'cv_summary.png'
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Summary plot saved: {summary_plot_path}")

    def _save_latex_table(self, statistics):
        latex_content = r"""\begin{table}[h]
\centering
\caption{""" + f"{self.n_splits}-Fold Cross-Validation Results" + r"""}
\begin{tabular}{lcccc}
\hline
\textbf{Scenario} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
\hline
"""

        for scenario_key, display_name in [
            ('test_set', 'Test Set'),
            ('white_background', 'White Background'),
            ('field_background', 'Field Background'),
            ('mixed', 'Mixed')
        ]:
            stats = statistics[scenario_key]
            latex_content += f"{display_name} & {stats['mean']:.4f} & {stats['std']:.4f} & {stats['min']:.4f} & {stats['max']:.4f} \\\\\n"

        latex_content += r"""\hline
\end{tabular}
\end{table}
"""

        latex_file = self.cv_dir / 'cv_results_table.tex'
        with open(latex_file, 'w') as f:
            f.write(latex_content)

        print(f"LaTeX table saved: {latex_file}")


def run_cross_validation(n_folds=5, epochs=100, batch_size=16, device=None):
    Config.ensure_directories()

    cv = KFoldCrossValidator(n_splits=n_folds, seed=Config.RANDOM_SEED)

    fold_results, statistics = cv.run_cross_validation(
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )

    print(f"\n{'='*80}")
    print(f"{n_folds}-Fold Cross-Validation Completed!")
    print(f"Results saved in: {cv.cv_dir}")
    print(f"{'='*80}\n")

    return fold_results, statistics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run K-Fold Cross-Validation')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs per fold')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    run_cross_validation(
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
