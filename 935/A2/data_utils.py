"""
Data processing utilities for rice leaf disease dataset
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import config


def find_folder_by_name(base_path: Path, target_name: str) -> str:
    """
    Find folder with matching name (case-insensitive, handles spaces)
    
    Args:
        base_path: Base directory to search in
        target_name: Name to search for
        
    Returns:
        Actual folder name found or original target_name
    """
    if not base_path.exists():
        return target_name
        
    try:
        items = list(base_path.iterdir())
        target_normalized = target_name.strip().lower()
        
        for item in items:
            if item.is_dir():
                if item.name.strip().lower() == target_normalized:
                    return item.name
                    
        return target_name
    except Exception:
        return target_name


def build_image_path(row: pd.Series) -> Tuple[str, str, str]:
    """
    Build full image path from CSV row data
    
    Args:
        row: Pandas Series containing CSV row data
        
    Returns:
        Tuple of (full_path, background, disease)
    """
    disease_str = row["Diseases"]
    filename = row["pictureName"]
    
    # Parse disease and background from Diseases field
    if "(" in disease_str:
        disease_folder = disease_str.split("(")[0].strip()
        background_part = disease_str.split("(")[1].replace(")", "").strip()
    else:
        disease_folder = disease_str.strip()
        background_part = "Unknown"
    
    # Apply background name corrections
    background_name = config.BACKGROUND_NAME_CORRECTIONS.get(background_part, background_part)
    
    # Find actual folder name (handles case variations and spaces)
    actual_background_folder = find_folder_by_name(config.BASE_DATA_PATH, background_name)
    
    # Build full path
    full_path = config.BASE_DATA_PATH / actual_background_folder / disease_folder / filename
    
    return str(full_path), background_part, disease_folder


def standardize_disease_name(disease_name: str) -> str:
    """Standardize disease name using correction mapping"""
    for old_name, new_name in config.DISEASE_NAME_CORRECTIONS.items():
        disease_name = disease_name.replace(old_name, new_name)
    return disease_name


def standardize_background_name(background_name: str) -> str:
    """Standardize background name using correction mapping"""
    for old_name, new_name in config.BACKGROUND_NAME_CORRECTIONS.items():
        background_name = background_name.replace(old_name, new_name)
    return background_name


def load_and_process_csv() -> pd.DataFrame:
    """
    Load CSV file and perform initial processing
    
    Returns:
        Processed DataFrame with additional columns
    """
    print(f"Loading CSV from: {config.CSV_FILE_PATH}")
    df = pd.read_csv(config.CSV_FILE_PATH)
    print(f"Loaded {len(df)} records")
    
    # Build image paths
    print("Building image paths...")
    path_data = df.apply(build_image_path, axis=1)
    df[["original_path", "original_background", "original_disease"]] = \
        pd.DataFrame(path_data.tolist(), index=df.index)
    
    # Standardize names
    print("Standardizing disease and background names...")
    df["disease_clean"] = df["original_disease"].apply(standardize_disease_name)
    df["background_clean"] = df["original_background"].apply(standardize_background_name)
    
    # Map to labels
    df["disease_label"] = df["disease_clean"].map(config.DISEASE_MAPPING)
    df["background_label"] = df["background_clean"].map(config.BACKGROUND_MAPPING)
    
    # Check for unmapped values
    unmapped_diseases = df[df["disease_label"].isna()]["disease_clean"].unique()
    unmapped_backgrounds = df[df["background_label"].isna()]["background_clean"].unique()
    
    if len(unmapped_diseases) > 0:
        print(f"Warning: Unmapped diseases found: {unmapped_diseases}")
    if len(unmapped_backgrounds) > 0:
        print(f"Warning: Unmapped backgrounds found: {unmapped_backgrounds}")
    
    # Add augmentation tracking columns
    df["is_augmented"] = False
    df["augmentation_type"] = "original"
    df["original_image"] = df["pictureName"]
    
    return df


def verify_image_files(df: pd.DataFrame, sample_size: int = 20) -> bool:
    """
    Verify that image files exist
    
    Args:
        df: DataFrame with image paths
        sample_size: Number of files to check
        
    Returns:
        True if verification passed, False otherwise
    """
    print(f"Verifying {sample_size} image files...")
    missing_count = 0
    
    for path in df["original_path"].head(sample_size):
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            missing_count += 1
    
    if missing_count == 0:
        print("All sampled files verified successfully")
        return True
    else:
        print(f"Warning: {missing_count} files missing out of {sample_size} checked")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'


def create_dataset_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create train/validation/test splits
    
    Args:
        df: DataFrame to split
        
    Returns:
        DataFrame with split assignments
    """
    from sklearn.model_selection import train_test_split
    
    df["split"] = "train"
    
    # Group by disease and background for stratified splitting
    groups = df.groupby(["disease_clean", "background_clean"])
    
    for (disease, background), group in groups:
        indices = group.index.tolist()
        
        # First split: train vs temp (validation + test)
        train_idx, temp_idx = train_test_split(
            indices, 
            test_size=(config.VAL_SPLIT_RATIO + config.TEST_SPLIT_RATIO),
            random_state=config.RANDOM_SEED
        )
        
        # Second split: validation vs test
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(config.TEST_SPLIT_RATIO / (config.VAL_SPLIT_RATIO + config.TEST_SPLIT_RATIO)),
            random_state=config.RANDOM_SEED
        )
        
        # Assign split labels
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "validation" 
        df.loc[test_idx, "split"] = "test"
    
    return df


def print_dataset_statistics(df: pd.DataFrame):
    """Print comprehensive dataset statistics"""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal images: {len(df)}")
    print(f"Original images: {len(df[~df['is_augmented']])}")
    print(f"Augmented images: {len(df[df['is_augmented']])}")
    
    print(f"\nSplit distribution:")
    print(df["split"].value_counts().sort_index())
    
    print(f"\nDisease distribution:")
    disease_counts = df.groupby("disease_clean").size().sort_values(ascending=False)
    for disease, count in disease_counts.items():
        print(f"  {disease}: {count}")
    
    print(f"\nBackground distribution:")
    background_counts = df.groupby("background_clean").size()
    for background, count in background_counts.items():
        print(f"  {background}: {count}")
    
    print(f"\nDisease-Background combinations:")
    combo_counts = df.groupby(["disease_clean", "background_clean"]).size()
    for (disease, background), count in combo_counts.items():
        print(f"  {disease} ({background}): {count}")
