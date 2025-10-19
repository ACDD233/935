import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import json


class DatasetPreprocessor:
    def __init__(self, source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        self.disease_mapping = {
            'Browon Spot': 'Brown_Spot',
            'Brown Spot': 'Brown_Spot',
            'Leaf Scaled': 'Leaf_Scald',
            'Rice Blast': 'Rice_Blast',
            'Rice Turgro': 'Rice_Tungro',
            'Rice Tungro': 'Rice_Tungro',
            'Sheath Blight': 'Sheath_Blight',
            'Shath Blight': 'Sheath_Blight'
        }

    def prepare_dataset(self):
        random.seed(self.seed)

        all_images = self._collect_images()

        splits = self._split_data(all_images)

        self._create_directory_structure(splits)

        stats = self._generate_statistics(splits)

        self._save_split_info(splits, stats)

        print(f"\nDataset preparation completed successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"\nDataset statistics:")
        print(f"Total images: {stats['total']}")
        print(f"Train: {stats['train']} ({self.train_ratio*100:.0f}%)")
        print(f"Validation: {stats['val']} ({self.val_ratio*100:.0f}%)")
        print(f"Test: {stats['test']} ({self.test_ratio*100:.0f}%)")

        return splits, stats

    def _collect_images(self):
        all_images = []

        for bg_type in ['Field Background', 'White Background']:
            bg_dir = self.source_dir / bg_type
            if not bg_dir.exists():
                continue

            for disease_dir in bg_dir.iterdir():
                if not disease_dir.is_dir():
                    continue

                disease_name = self.disease_mapping.get(disease_dir.name, disease_dir.name)

                for img_file in disease_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        all_images.append({
                            'path': img_file,
                            'disease': disease_name,
                            'background': bg_type.replace(' ', '_').lower()
                        })

        print(f"Collected {len(all_images)} images from dataset")
        return all_images

    def _split_data(self, all_images):
        images_by_disease = {}
        for img in all_images:
            disease = img['disease']
            if disease not in images_by_disease:
                images_by_disease[disease] = []
            images_by_disease[disease].append(img)

        splits = {'train': [], 'val': [], 'test': []}

        for disease, images in images_by_disease.items():
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            splits['train'].extend(images[:n_train])
            splits['val'].extend(images[n_train:n_train + n_val])
            splits['test'].extend(images[n_train + n_val:])

            print(f"{disease}: {n_total} images -> Train: {n_train}, Val: {n_val}, Test: {n_total - n_train - n_val}")

        for split in splits.values():
            random.shuffle(split)

        return splits

    def _create_directory_structure(self, splits):
        for split_name, images in splits.items():
            for img_info in images:
                disease = img_info['disease']
                dest_dir = self.output_dir / split_name / disease
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_path = dest_dir / img_info['path'].name
                shutil.copy2(img_info['path'], dest_path)

    def _generate_statistics(self, splits):
        stats = {
            'total': sum(len(images) for images in splits.values()),
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test']),
            'by_disease': {},
            'by_background': {}
        }

        for split_name, images in splits.items():
            for img_info in images:
                disease = img_info['disease']
                bg = img_info['background']

                if disease not in stats['by_disease']:
                    stats['by_disease'][disease] = {'train': 0, 'val': 0, 'test': 0, 'total': 0}
                stats['by_disease'][disease][split_name] += 1
                stats['by_disease'][disease]['total'] += 1

                if bg not in stats['by_background']:
                    stats['by_background'][bg] = {'train': 0, 'val': 0, 'test': 0, 'total': 0}
                stats['by_background'][bg][split_name] += 1
                stats['by_background'][bg]['total'] += 1

        return stats

    def _save_split_info(self, splits, stats):
        info = {
            'seed': self.seed,
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'statistics': stats,
            'disease_classes': list(stats['by_disease'].keys())
        }

        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\nDataset information saved to {info_path}")


def prepare_dataset_for_yolo(source_dir='./Dhan-Shomadhan', output_dir=None, seed=42):
    if output_dir is None:
        output_dir = Path(source_dir) / 'processed_data'

    preprocessor = DatasetPreprocessor(
        source_dir=source_dir,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )

    return preprocessor.prepare_dataset()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataset for rice disease classification')
    parser.add_argument('--source', type=str, default='./Dhan-Shomadhan',
                        help='Source dataset directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: <source>/processed_data)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    output = args.output if args.output else str(Path(args.source) / 'processed_data')
    prepare_dataset_for_yolo(args.source, output, args.seed)
