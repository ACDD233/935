import os
from pathlib import Path
import torch


class Config:
    BASE_DIR = Path(__file__).parent
    DATASET_DIR = BASE_DIR / 'Dhan-Shomadhan'
    PROCESSED_DIR = DATASET_DIR / 'processed_data'
    RESULTS_DIR = BASE_DIR / 'results'
    MODELS_DIR = BASE_DIR / 'models'

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42

    YOLO_MODEL = 'yolov8s-cls.pt'
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 100
    PATIENCE = 20

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    CLASS_NAMES = ['Brown_Spot', 'Leaf_Scald', 'Rice_Blast', 'Rice_Tungro', 'Sheath_Blight']

    SAVE_PERIOD = 10

    @classmethod
    def ensure_directories(cls):
        for dir_path in [cls.PROCESSED_DIR, cls.RESULTS_DIR, cls.MODELS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_data_yaml_path(cls):
        return cls.PROCESSED_DIR / 'data.yaml'


if __name__ == '__main__':
    Config.ensure_directories()
    print(f"Device: {Config.DEVICE}")
    print(f"Dataset directory: {Config.DATASET_DIR}")
    print(f"Results directory: {Config.RESULTS_DIR}")
