"""
Configuration settings for rice leaf disease data augmentation pipeline
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
BASE_DATA_PATH = PROJECT_ROOT / "Dhan-Shomadhan"
OUTPUT_DATA_PATH = PROJECT_ROOT / "enhanced_data_windows"
CSV_FILE_PATH = BASE_DATA_PATH / "Dhan-Shomadhan_picture_Information.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "cleaned_augmented_data_windows.csv"

# Data processing parameters
TARGET_IMAGES_PER_CLASS = 1000
RANDOM_SEED = 42

# Image augmentation parameters
AUGMENTATION_TYPES = [
    "rotate", "flip_h", "flip_v", "scale", 
    "blur", "brightness", "contrast", "noise", "combo"
]

# Disease and background mappings
DISEASE_MAPPING = {
    "Brown Spot": 0,
    "Leaf Scald": 1,
    "Rice Blast": 2,
    "Rice Tungro": 3,
    "Sheath Blight": 4,
}

BACKGROUND_MAPPING = {
    "Field Background": 0, 
    "White Background": 1
}

# Name standardization mappings
DISEASE_NAME_CORRECTIONS = {
    "Browon Spot": "Brown Spot",
    "Shath Blight": "Sheath Blight", 
    "Rice Turgro": "Rice Tungro",
    "Leaf Scaled": "Leaf Scald"
}

BACKGROUND_NAME_CORRECTIONS = {
    "Feild Background": "Field Background",
    "white Background": "White Background"
}

# Augmentation parameter ranges
AUGMENTATION_PARAMS = {
    "rotate": {"min_angle": -30, "max_angle": 30},
    "scale": {"min_scale": 0.85, "max_scale": 0.95},
    "blur": {"kernel_sizes": [3, 5, 7]},
    "brightness": {"min_factor": 0.7, "max_factor": 1.3},
    "contrast": {"min_factor": 0.8, "max_factor": 1.2},
    "noise": {"min_level": 5, "max_level": 15}
}

# Train/validation/test split ratios
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# Verify split ratios sum to 1.0
assert abs(TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO + TEST_SPLIT_RATIO - 1.0) < 1e-6, \
    "Split ratios must sum to 1.0"
