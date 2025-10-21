"""
Configuration management and command-line argument parsing
"""
import argparse
import json
import os
from datetime import datetime


def detect_background_folders(source_dir='./Dhan-Shomadhan'):
    """
    Auto-detect Field Background and White Background folder names
    Handles cross-platform spacing differences

    Returns:
        tuple: (field_folder, white_folder)
    """
    field_folder = None
    white_folder = None

    if not os.path.exists(source_dir):
        return None, None

    items = os.listdir(source_dir)

    for item in items:
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            item_lower = item.lower()

            if 'field' in item_lower and 'background' in item_lower:
                field_folder = item
            elif 'white' in item_lower and 'background' in item_lower:
                white_folder = item

    return field_folder, white_folder


class Config:
    """Configuration class for all training, testing and visualization parameters"""

    def __init__(self):
        self.source_data_dir = './Dhan-Shomadhan'

        # Auto-detect background folders
        field_folder, white_folder = detect_background_folders(self.source_data_dir)

        if field_folder and white_folder:
            self.background_folders = [field_folder, white_folder]
        else:
            self.background_folders = ['Field Background  ', 'White Background ']
            if not field_folder or not white_folder:
                print(f"Warning: Background folders not found, using defaults")

        # Disease name mapping (correct spelling errors)
        self.disease_name_mapping = {
            'Browon Spot': 'Brown_Spot',
            'Brown Spot': 'Brown_Spot',
            'Leaf Scaled': 'Leaf_Scaled',
            'Rice Blast': 'Rice_Blast',
            'Rice Turgro': 'Rice_Tungro',
            'Rice Tungro': 'Rice_Tungro',
            'Sheath Blight': 'Sheath_Blight',
            'Shath Blight': 'Sheath_Blight'
        }

        self.random_seed = 42
        self.output_base_dir = None
        self.n_splits = 5

        # Model configuration
        self.model_name = 'yolo11s-cls.pt'
        self.epochs = 150
        self.imgsz = 320
        self.batch_size = 16
        self.device = 0
        self.freeze = 0
        self.patience = 30
        self.optimizer = 'AdamW'
        self.lr0 = 0.0005
        self.weight_decay = 0.0005
        self.momentum = 0.937
        self.warmup_epochs = 3
        self.warmup_momentum = 0.8
        self.warmup_bias_lr = 0.1

        # Data augmentation
        self.augment = True
        self.degrees = 10
        self.translate = 0.05
        self.scale = 0.05
        self.fliplr = 0.3
        self.flipud = 0.0
        self.mosaic = 0.0
        self.mixup = 0.0

        # Training configuration
        self.cache = False
        self.workers = 8
        self.verbose = True
        self.save = True
        self.save_period = -1
        self.plots = True

        # Run mode
        self.mode = 'all'

        # Visualization
        self.vis_num_samples = 10
        self.vis_target_layer = None

    def set_output_dir(self):
        """Set output directory based on random seed"""
        self.output_base_dir = f'./Dhan-Shomadhan/{self.random_seed}'
        self.dataset_dir = os.path.join(self.output_base_dir, 'datasets')
        self.results_dir = os.path.join(self.output_base_dir, 'results')
        self.models_dir = os.path.join(self.output_base_dir, 'models')
        self.visualizations_dir = os.path.join(self.output_base_dir, 'visualizations')
        self.logs_dir = os.path.join(self.output_base_dir, 'logs')

    def to_dict(self):
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save_config(self, filepath):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        config_dict = self.to_dict()
        config_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath):
        """Load configuration from JSON file"""
        config = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if key != 'timestamp':
                setattr(config, key, value)
        return config


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='YOLO K-Fold Cross-Validation Training System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--mode', type=str, default='all',
                        choices=['split', 'train', 'test', 'visualize', 'all'],
                        help='Run mode')
    parser.add_argument('--source_data_dir', type=str, default='./Dhan-Shomadhan',
                        help='Source data directory')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='yolo11s-cls.pt')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--freeze', type=int, default=0)

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'])
    parser.add_argument('--lr0', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--warmup_epochs', type=float, default=3.0)
    parser.add_argument('--warmup_momentum', type=float, default=0.8)
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1)

    # Augmentation parameters
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no_augment', action='store_false', dest='augment')
    parser.add_argument('--degrees', type=float, default=10.0)
    parser.add_argument('--translate', type=float, default=0.05)
    parser.add_argument('--scale', type=float, default=0.05)
    parser.add_argument('--fliplr', type=float, default=0.3)
    parser.add_argument('--flipud', type=float, default=0.0)
    parser.add_argument('--mosaic', type=float, default=0.0)
    parser.add_argument('--mixup', type=float, default=0.0)

    # Training configuration
    parser.add_argument('--cache', action='store_true', default=False)
    parser.add_argument('--no_cache', action='store_false', dest='cache')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--plots', action='store_true', default=True)

    # Visualization
    parser.add_argument('--vis_num_samples', type=int, default=10)
    parser.add_argument('--vis_target_layer', type=str, default=None)

    parser.add_argument('--load_config', type=str, default=None,
                        help='Load config from JSON file')

    args = parser.parse_args()

    if args.load_config:
        config = Config.from_json(args.load_config)
    else:
        config = Config()

    for key, value in vars(args).items():
        if key != 'load_config' and value is not None:
            setattr(config, key, value)

    config.set_output_dir()
    return config


if __name__ == '__main__':
    config = parse_args()
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
