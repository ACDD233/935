"""
Main data augmentation pipeline for rice leaf disease dataset
"""

import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

import config
import data_utils
from image_augmentation import ImageAugmentor


class DataAugmentationPipeline:
    """Main pipeline class for data augmentation workflow"""
    
    def __init__(self):
        """Initialize pipeline with augmentor and configuration"""
        self.augmentor = ImageAugmentor()
        self.output_path = config.OUTPUT_DATA_PATH
        self.augmentation_types = self.augmentor.get_augmentation_types()
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        random.seed(config.RANDOM_SEED)
    
    def process_dataset(self):
        """Main method to process the entire dataset"""
        print("=" * 60)
        print("Rice Leaf Disease Data Augmentation Pipeline")
        print("=" * 60)
        
        # Step 1: Load and process data
        df = self._load_and_prepare_data()
        
        # Step 2: Perform augmentation
        augmented_data = self._augment_dataset(df)
        
        # Step 3: Merge and split data
        final_df = self._merge_and_split_data(df, augmented_data)
        
        # Step 4: Save results
        self._save_results(final_df)
        
        # Step 5: Print statistics
        data_utils.print_dataset_statistics(final_df)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
    
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load CSV data and perform initial processing"""
        print("\n[Step 1] Loading and preparing data...")
        print("-" * 40)
        
        # Load and process CSV
        df = data_utils.load_and_process_csv()
        
        # Verify image files exist
        if not data_utils.verify_image_files(df):
            raise RuntimeError("Image file verification failed")
        
        # Print data distribution
        print(f"\nData distribution by disease and background:")
        print(df.groupby(["disease_clean", "background_clean"]).size())
        
        return df
    
    def _augment_dataset(self, df: pd.DataFrame) -> list:
        """Perform data augmentation on the dataset"""
        print("\n[Step 2] Performing data augmentation...")
        print("-" * 40)
        
        augmented_records = []
        groups = df.groupby(["disease_clean", "background_clean"])
        
        for (disease, background), group in groups:
            print(f"\nProcessing: {disease} ({background})")
            
            current_count = len(group)
            needed = config.TARGET_IMAGES_PER_CLASS - current_count
            
            if needed <= 0:
                print(f"  Already has {current_count} images, skipping")
                continue
            
            print(f"  Current: {current_count}, Need: {needed} more")
            
            # Create output directory
            output_dir = self.output_path / background / disease
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate augmented images
            new_records = self._augment_class_group(
                group, output_dir, disease, background, needed
            )
            augmented_records.extend(new_records)
        
        print(f"\nGenerated {len(augmented_records)} augmented images")
        return augmented_records
    
    def _augment_class_group(self, group: pd.DataFrame, output_dir: Path, 
                           disease: str, background: str, needed: int) -> list:
        """Augment images for a specific disease-background combination"""
        records = []
        used_combinations = set()
        
        original_images = group["original_path"].tolist()
        original_names = group["pictureName"].tolist()
        
        # Phase 1: Systematic augmentation (each type applied once per image)
        phase1_limit = min(needed, len(original_images) * len(self.augmentation_types))
        img_idx = 0
        type_idx = 0
        aug_count = 0
        
        with tqdm(total=needed, desc=f"  Augmenting {disease}") as pbar:
            # Phase 1: Systematic approach
            while aug_count < phase1_limit:
                if self._process_single_augmentation(
                    original_images, original_names, output_dir, 
                    disease, background, group, img_idx, type_idx, 
                    used_combinations, records, aug_count
                ):
                    aug_count += 1
                    pbar.update(1)
                
                # Move to next combination
                type_idx = (type_idx + 1) % len(self.augmentation_types)
                if type_idx == 0:
                    img_idx = (img_idx + 1) % len(original_images)
            
            # Phase 2: Random augmentation if still needed
            max_attempts = needed * 10
            attempts = 0
            
            while aug_count < needed and attempts < max_attempts:
                attempts += 1
                
                # Random selection
                img_idx = random.randint(0, len(original_images) - 1)
                aug_type = random.choice(self.augmentation_types)
                
                # Skip deterministic augmentations if already used
                if aug_type in ["flip_h", "flip_v"]:
                    combo_key = (original_names[img_idx], aug_type)
                    if combo_key in used_combinations:
                        continue
                    used_combinations.add(combo_key)
                
                if self._process_single_augmentation(
                    original_images, original_names, output_dir,
                    disease, background, group, img_idx, aug_type,
                    used_combinations, records, aug_count
                ):
                    aug_count += 1
                    pbar.update(1)
        
        return records
    
    def _process_single_augmentation(self, original_images: list, original_names: list,
                                   output_dir: Path, disease: str, background: str,
                                   group: pd.DataFrame, img_idx: int, aug_type_or_idx,
                                   used_combinations: set, records: list, aug_count: int) -> bool:
        """Process a single image augmentation"""
        try:
            # Handle both index and direct type
            if isinstance(aug_type_or_idx, int):
                aug_type = self.augmentation_types[aug_type_or_idx]
            else:
                aug_type = aug_type_or_idx
            
            original_path = original_images[img_idx]
            original_name = original_names[img_idx]
            
            # Check for duplicate combinations
            combo_key = (original_name, aug_type)
            if combo_key in used_combinations:
                return False
            
            # Verify file exists and load image
            if not os.path.exists(original_path):
                return False
            
            image = cv2.imread(original_path)
            if image is None:
                return False
            
            # Apply augmentation
            augmented_img, param_desc = self.augmentor.augment_image(image, aug_type)
            
            # Generate output filename
            base_name = Path(original_name).stem
            ext = Path(original_name).suffix
            new_filename = f"{base_name}_aug{aug_count:04d}_{aug_type}{ext}"
            output_path = output_dir / new_filename
            
            # Save augmented image
            if cv2.imwrite(str(output_path), augmented_img):
                used_combinations.add(combo_key)
                
                # Create record
                original_row = group[group["pictureName"] == original_name].iloc[0]
                record = {
                    "pictureName": new_filename,
                    "Diseases": f"{disease}({background})",
                    "original_path": str(output_path),
                    "original_background": background,
                    "original_disease": disease,
                    "disease_clean": disease,
                    "background_clean": background,
                    "disease_label": original_row["disease_label"],
                    "background_label": original_row["background_label"],
                    "is_augmented": True,
                    "augmentation_type": f"{aug_type}({param_desc})",
                    "original_image": original_name,
                }
                records.append(record)
                return True
            
        except Exception as e:
            print(f"Error processing augmentation: {e}")
        
        return False
    
    def _merge_and_split_data(self, original_df: pd.DataFrame, 
                            augmented_records: list) -> pd.DataFrame:
        """Merge original and augmented data, then create splits"""
        print("\n[Step 3] Merging data and creating splits...")
        print("-" * 40)
        
        # Merge data
        augmented_df = pd.DataFrame(augmented_records)
        final_df = pd.concat([original_df, augmented_df], ignore_index=True)
        
        # Create splits
        final_df = data_utils.create_dataset_splits(final_df)
        
        return final_df
    
    def _save_results(self, df: pd.DataFrame):
        """Save final dataset to CSV"""
        print("\n[Step 4] Saving results...")
        print("-" * 40)
        
        # Save to CSV
        df.to_csv(config.OUTPUT_CSV_PATH, index=False)
        print(f"Dataset saved to: {config.OUTPUT_CSV_PATH}")
        
        # Print split summary
        print(f"\nSplit distribution:")
        split_counts = df["split"].value_counts().sort_index()
        for split, count in split_counts.items():
            print(f"  {split}: {count}")


def main():
    """Main entry point for the pipeline"""
    try:
        pipeline = DataAugmentationPipeline()
        pipeline.process_dataset()
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
