"""
Experiment logging module
"""
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """Experiment logger for saving configurations and results"""

    def __init__(self, config):
        self.config = config
        self.logs_dir = config.logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

        self.experiment_log_path = os.path.join(self.logs_dir, 'experiment_log.json')
        self.summary_csv_path = os.path.join(self.logs_dir, 'experiments_summary.csv')

        self.experiment_id = self.generate_experiment_id()

    def generate_experiment_id(self):
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"exp_{timestamp}_seed{self.config.random_seed}"

    def log_experiment_start(self):
        """Log experiment start"""
        print(f"\nExperiment: {self.experiment_id}")
        print(f"Seed: {self.config.random_seed}, Mode: {self.config.mode}\n")

        config_path = os.path.join(self.logs_dir, f'{self.experiment_id}_config.json')
        self.config.save_config(config_path)

    def log_experiment_results(self, split_info=None, train_results=None, test_results=None):
        """Log experiment results"""
        experiment_record = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'random_seed': self.config.random_seed,
            'mode': self.config.mode,
            'config': self.config.to_dict(),
        }

        if split_info:
            experiment_record['split_info'] = {
                'total_images': split_info.get('total_images'),
                'class_names': split_info.get('class_names'),
                'n_splits': split_info.get('n_splits'),
            }

        if train_results:
            train_summary_path = os.path.join(self.config.results_dir, 'training_summary.json')
            if os.path.exists(train_summary_path):
                with open(train_summary_path, 'r', encoding='utf-8') as f:
                    train_summary = json.load(f)
                experiment_record['training'] = {
                    'top1_mean': train_summary.get('top1_mean'),
                    'top1_std': train_summary.get('top1_std'),
                    'top1_min': train_summary.get('top1_min'),
                    'top1_max': train_summary.get('top1_max'),
                }

        if test_results:
            test_summary_path = os.path.join(self.config.results_dir, 'test_results', 'test_summary.json')
            if os.path.exists(test_summary_path):
                with open(test_summary_path, 'r', encoding='utf-8') as f:
                    test_summary = json.load(f)
                experiment_record['testing'] = {
                    'top1_mean': test_summary.get('top1_mean'),
                    'top1_std': test_summary.get('top1_std'),
                    'top1_min': test_summary.get('top1_min'),
                    'top1_max': test_summary.get('top1_max'),
                }

        experiment_file = os.path.join(self.logs_dir, f'{self.experiment_id}_results.json')
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_record, f, indent=4, ensure_ascii=False)

        self.append_to_experiment_log(experiment_record)
        self.update_summary_table(experiment_record)

    def append_to_experiment_log(self, record):
        """Append to experiment log"""
        if os.path.exists(self.experiment_log_path):
            with open(self.experiment_log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(record)

        with open(self.experiment_log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

    def update_summary_table(self, record):
        """Update experiment summary table"""
        summary_row = {
            'experiment_id': record['experiment_id'],
            'timestamp': record['timestamp'],
            'random_seed': record['random_seed'],
            'mode': record['mode'],
            'model_name': record['config'].get('model_name'),
            'epochs': record['config'].get('epochs'),
            'batch_size': record['config'].get('batch_size'),
            'lr0': record['config'].get('lr0'),
            'optimizer': record['config'].get('optimizer'),
            'augment': record['config'].get('augment'),
        }

        if 'training' in record:
            summary_row.update({
                'train_top1_mean': record['training'].get('top1_mean'),
                'train_top1_std': record['training'].get('top1_std'),
                'train_top1_max': record['training'].get('top1_max'),
            })

        if 'testing' in record:
            summary_row.update({
                'test_top1_mean': record['testing'].get('top1_mean'),
                'test_top1_std': record['testing'].get('top1_std'),
                'test_top1_max': record['testing'].get('top1_max'),
            })

        if os.path.exists(self.summary_csv_path):
            df = pd.read_csv(self.summary_csv_path, encoding='utf-8-sig')
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            df = pd.DataFrame([summary_row])

        df.to_csv(self.summary_csv_path, index=False, encoding='utf-8-sig')

    def print_summary_table(self):
        """Print experiment summary table"""
        if os.path.exists(self.summary_csv_path):
            df = pd.read_csv(self.summary_csv_path, encoding='utf-8-sig')
            print("\nExperiment Summary:")
            print(df.to_string(index=False))
        else:
            print("No experiments recorded")


def create_experiment_logger(config):
    """Create experiment logger"""
    return ExperimentLogger(config)


if __name__ == '__main__':
    from config import parse_args

    config = parse_args()
    logger = create_experiment_logger(config)
    logger.log_experiment_start()
    logger.print_summary_table()
