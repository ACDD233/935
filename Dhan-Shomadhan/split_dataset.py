"""
Dataset splitting module using StratifiedKFold for K-Fold cross-validation
Ensures reproducible splits with the same random seed
"""
import os
import glob
import shutil
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


class DatasetSplitter:
    """Dataset splitter with reproducible K-Fold splitting"""

    def __init__(self, config):
        self.config = config
        self.source_data_dir = config.source_data_dir
        self.background_folders = config.background_folders
        self.disease_name_mapping = config.disease_name_mapping
        self.n_splits = config.n_splits
        self.random_seed = config.random_seed
        self.dataset_dir = config.dataset_dir

        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        self.disease_images = {}

    def collect_images(self):
        """Collect images from Field Background and White Background folders"""
        print(f"\nCollecting images (seed={self.random_seed})...")

        if not os.path.exists(self.source_data_dir):
            raise FileNotFoundError(f"Source directory not found: {self.source_data_dir}")

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif',
                            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.GIF']

        # Process each background folder
        for bg_folder in self.background_folders:
            bg_path = os.path.join(self.source_data_dir, bg_folder)

            if not os.path.exists(bg_path):
                print(f"Warning: Folder not found - {bg_folder}")
                continue

            for disease_folder in os.listdir(bg_path):
                disease_path = os.path.join(bg_path, disease_folder)

                if not os.path.isdir(disease_path):
                    continue

                # Correct disease name
                normalized_name = self.disease_name_mapping.get(disease_folder, disease_folder)

                # Collect images
                disease_images = []
                for ext in image_extensions:
                    pattern = os.path.join(disease_path, ext)
                    disease_images.extend(glob.glob(pattern))

                if normalized_name not in self.disease_images:
                    self.disease_images[normalized_name] = []

                self.disease_images[normalized_name].extend(disease_images)

        # Build final class list
        self.class_names = sorted(self.disease_images.keys())
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        print(f"Found {len(self.class_names)} classes:")
        for class_name in self.class_names:
            images = self.disease_images[class_name]
            class_idx = self.class_to_idx[class_name]
            self.image_paths.extend(images)
            self.labels.extend([class_idx] * len(images))
            print(f"  {class_name}: {len(images)} images")

        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.source_data_dir}")

        print(f"Total: {len(self.image_paths)} images\n")
        return self.image_paths, self.labels

    def split_and_save(self):
        """Split dataset using StratifiedKFold and save to filesystem"""
        if len(self.image_paths) == 0:
            self.collect_images()

        print(f"Starting {self.n_splits}-Fold split...")

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_seed
        )

        split_info = {
            'random_seed': self.random_seed,
            'n_splits': self.n_splits,
            'total_images': len(self.image_paths),
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'folds': []
        }

        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(self.image_paths, self.labels)):
            fold_num = fold_idx + 1
            print(f"  Fold {fold_num}: {len(train_indices)} train, {len(val_indices)} val")

            fold_dir = os.path.join(self.dataset_dir, f'fold_{fold_num}')
            train_dir = os.path.join(fold_dir, 'train')
            val_dir = os.path.join(fold_dir, 'val')

            if os.path.exists(fold_dir):
                shutil.rmtree(fold_dir)

            for split_name in ['train', 'val']:
                for class_name in self.class_names:
                    class_dir = os.path.join(fold_dir, split_name, class_name)
                    os.makedirs(class_dir, exist_ok=True)

            # Copy train images
            for idx in train_indices:
                src_path = self.image_paths[idx]
                label_name = self.class_names[self.labels[idx]]
                dst_path = os.path.join(train_dir, label_name, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)

            # Copy val images
            for idx in val_indices:
                src_path = self.image_paths[idx]
                label_name = self.class_names[self.labels[idx]]
                dst_path = os.path.join(val_dir, label_name, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)

            fold_info = {
                'fold': fold_num,
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'train_indices': train_indices.tolist(),
                'val_indices': val_indices.tolist()
            }
            split_info['folds'].append(fold_info)

        # Save split info
        split_info_path = os.path.join(self.dataset_dir, 'split_info.json')
        with open(split_info_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=4, ensure_ascii=False)

        print(f"Split complete. Saved to {self.dataset_dir}\n")
        return split_info


def split_dataset(config):
    """Main function for dataset splitting"""
    splitter = DatasetSplitter(config)
    split_info = splitter.split_and_save()
    return split_info


if __name__ == '__main__':
    from config import parse_args
    config = parse_args()
    split_info = split_dataset(config)
