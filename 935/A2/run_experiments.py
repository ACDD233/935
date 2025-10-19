import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from config import Config
from data_preprocessor import prepare_dataset_for_yolo
from trainer import train_model
from evaluator import ModelEvaluator


class ExperimentRunner:
    def __init__(self, n_runs=5, base_seed=42):
        self.n_runs = n_runs
        self.base_seed = base_seed
        self.results = []

        self.experiment_dir = Config.RESULTS_DIR / f'experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def run_multiple_experiments(self, epochs=100, batch_size=16, device=None):
        print("\n" + "="*80)
        print(f"Running {self.n_runs} independent experiments with different data splits")
        print("="*80)

        for run_idx in range(self.n_runs):
            seed = self.base_seed + run_idx

            print(f"\n{'='*80}")
            print(f"Experiment {run_idx + 1}/{self.n_runs} (seed: {seed})")
            print(f"{'='*80}")

            run_result = self._run_single_experiment(run_idx, seed, epochs, batch_size, device)
            self.results.append(run_result)

            self._save_intermediate_results()

        self._compute_and_save_statistics()

        return self.results

    def _run_single_experiment(self, run_idx, seed, epochs, batch_size, device):
        Config.RANDOM_SEED = seed

        print(f"\n[Run {run_idx + 1}] Preparing dataset with seed {seed}...")
        dataset_dir = Config.PROCESSED_DIR.parent / f'dataset_run_{run_idx}'
        splits, stats = prepare_dataset_for_yolo(
            source_dir=str(Config.DATASET_DIR),
            output_dir=str(dataset_dir),
            seed=seed
        )

        original_processed_dir = Config.PROCESSED_DIR
        Config.PROCESSED_DIR = dataset_dir

        print(f"\n[Run {run_idx + 1}] Training model...")
        trainer, train_results = train_model(
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )

        best_model_path = trainer.get_best_model_path()

        print(f"\n[Run {run_idx + 1}] Evaluating model on scenarios...")
        evaluator = ModelEvaluator(best_model_path, device=device)

        scenario_results = evaluator.evaluate_scenarios(
            save_dir=self.experiment_dir / f'run_{run_idx}_scenarios'
        )

        test_results = evaluator.evaluate_on_split(
            split='test',
            save_dir=self.experiment_dir / f'run_{run_idx}_test'
        )

        Config.PROCESSED_DIR = original_processed_dir

        run_result = {
            'run_idx': run_idx,
            'seed': seed,
            'test_accuracy': test_results.get('accuracy', 0.0),
            'scenarios': {
                'white_background': scenario_results.get('white_background', {}).get('accuracy', 0.0),
                'field_background': scenario_results.get('field_background', {}).get('accuracy', 0.0),
                'mixed': scenario_results.get('mixed', {}).get('accuracy', 0.0)
            }
        }

        print(f"\n[Run {run_idx + 1}] Results:")
        print(f"  Test accuracy: {run_result['test_accuracy']:.4f}")
        print(f"  White background: {run_result['scenarios']['white_background']:.4f}")
        print(f"  Field background: {run_result['scenarios']['field_background']:.4f}")
        print(f"  Mixed: {run_result['scenarios']['mixed']:.4f}")

        return run_result

    def _save_intermediate_results(self):
        results_file = self.experiment_dir / 'intermediate_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _compute_and_save_statistics(self):
        print("\n" + "="*80)
        print("Computing statistics across all runs")
        print("="*80)

        test_accs = [r['test_accuracy'] for r in self.results]
        white_accs = [r['scenarios']['white_background'] for r in self.results]
        field_accs = [r['scenarios']['field_background'] for r in self.results]
        mixed_accs = [r['scenarios']['mixed'] for r in self.results]

        statistics = {
            'n_runs': self.n_runs,
            'test_set': {
                'mean': float(np.mean(test_accs)),
                'std': float(np.std(test_accs)),
                'min': float(np.min(test_accs)),
                'max': float(np.max(test_accs))
            },
            'white_background': {
                'mean': float(np.mean(white_accs)),
                'std': float(np.std(white_accs)),
                'min': float(np.min(white_accs)),
                'max': float(np.max(white_accs))
            },
            'field_background': {
                'mean': float(np.mean(field_accs)),
                'std': float(np.std(field_accs)),
                'min': float(np.min(field_accs)),
                'max': float(np.max(field_accs))
            },
            'mixed': {
                'mean': float(np.mean(mixed_accs)),
                'std': float(np.std(mixed_accs)),
                'min': float(np.min(mixed_accs)),
                'max': float(np.max(mixed_accs))
            }
        }

        stats_file = self.experiment_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)

        print("\nFinal Statistics (Mean ± Std):")
        print(f"  Test set:         {statistics['test_set']['mean']:.4f} ± {statistics['test_set']['std']:.4f}")
        print(f"  White background: {statistics['white_background']['mean']:.4f} ± {statistics['white_background']['std']:.4f}")
        print(f"  Field background: {statistics['field_background']['mean']:.4f} ± {statistics['field_background']['std']:.4f}")
        print(f"  Mixed:            {statistics['mixed']['mean']:.4f} ± {statistics['mixed']['std']:.4f}")

        self._plot_statistics(statistics)

        self._save_latex_table(statistics)

        return statistics

    def _plot_statistics(self, statistics):
        scenarios = ['Test Set', 'White BG', 'Field BG', 'Mixed']
        means = [
            statistics['test_set']['mean'],
            statistics['white_background']['mean'],
            statistics['field_background']['mean'],
            statistics['mixed']['mean']
        ]
        stds = [
            statistics['test_set']['std'],
            statistics['white_background']['std'],
            statistics['field_background']['std'],
            statistics['mixed']['std']
        ]

        fig, ax = plt.subplots(figsize=(12, 7))

        x_pos = np.arange(len(scenarios))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color='#3498db',
                     edgecolor='black', linewidth=1.5, alpha=0.8)

        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Evaluation Scenario', fontsize=14, fontweight='bold')
        ax.set_title(f'Model Performance across {self.n_runs} Independent Runs\n(Mean ± Standard Deviation)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()

        plot_path = self.experiment_dir / 'overall_statistics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nStatistics plot saved: {plot_path}")

    def _save_latex_table(self, statistics):
        latex_content = r"""\begin{table}[h]
\centering
\caption{Model Performance across 5 Independent Runs}
\begin{tabular}{lcccc}
\hline
\textbf{Scenario} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
\hline
"""

        for scenario_name, display_name in [
            ('test_set', 'Test Set'),
            ('white_background', 'White Background'),
            ('field_background', 'Field Background'),
            ('mixed', 'Mixed')
        ]:
            stats = statistics[scenario_name]
            latex_content += f"{display_name} & {stats['mean']:.4f} & {stats['std']:.4f} & {stats['min']:.4f} & {stats['max']:.4f} \\\\\n"

        latex_content += r"""\hline
\end{tabular}
\end{table}
"""

        latex_file = self.experiment_dir / 'results_table.tex'
        with open(latex_file, 'w') as f:
            f.write(latex_content)

        print(f"LaTeX table saved: {latex_file}")


def run_experiments(n_runs=5, epochs=100, batch_size=16, device=None):
    Config.ensure_directories()

    runner = ExperimentRunner(n_runs=n_runs)
    results = runner.run_multiple_experiments(epochs=epochs, batch_size=batch_size, device=device)

    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print(f"Results saved in: {runner.experiment_dir}")
    print("="*80)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run multiple experiments with different data splits')
    parser.add_argument('--n-runs', type=int, default=5,
                       help='Number of independent runs (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs per run')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    run_experiments(
        n_runs=args.n_runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
