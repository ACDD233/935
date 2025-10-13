"""
Rice Leaf Disease Dataset Organization and Enhancement Script - Improved Version
Improvements:
1. Added randomness to all augmentation methods
2. Track used combinations to avoid duplicates
3. Priority strategy: Apply each augmentation type once per image first
"""

import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ===== Configuration Parameters =====
BASE_PATH = "./Dhan-Shomadhan/"
OUTPUT_PATH = "./enhanced_data"
CSV_PATH = "./Dhan-Shomadhan/Dhan-Shomadhan_picture_Information.csv"
OUTPUT_CSV = "./cleaned_augmented_data.csv"
TARGET_IMAGES_PER_CLASS = 1000
RANDOM_SEED = 42

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("Rice Leaf Disease Dataset Enhancement Script (Improved)")
print("=" * 60)

# ===== Step 1: Read and Analyze CSV =====
print("\n[Step 1] Read and Analyze CSV")
print("-" * 60)

df = pd.read_csv(CSV_PATH)
print(f"✓ Read Completed: Total {len(df)} records")

print("\nSample of Diseases field in Original CSV:")
print(df["Diseases"].value_counts().head(10))

# ===== Step 2: Build Correct Image Path =====
print("\n[Step 2] Build Correct Image Path")
print("-" * 60)


def find_matching_folder(base_path, target_name):
    """
    Dynamically find matching folder name (case-insensitive, handles trailing spaces)
    This works cross-platform on Windows, Mac, and Linux
    """
    try:
        # List all items in the directory
        items = os.listdir(base_path)
        # Normalize the target name for comparison
        target_normalized = target_name.strip().lower()

        for item in items:
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Compare normalized names
                if item.strip().lower() == target_normalized:
                    return item  # Return the actual folder name (with spaces if any)

        # If no match found, return the target as-is
        return target_name
    except Exception:
        return target_name


def build_original_path(row):
    """Build path using original information from CSV, mapping to actual folder names"""
    disease_str = row["Diseases"]
    filename = row["pictureName"]

    if "(" in disease_str:
        disease_folder = disease_str.split("(")[0].strip()
        background_part = disease_str.split("(")[1].replace(")", "").strip()
    else:
        disease_folder = disease_str.strip()
        background_part = "Unknown"

    # Map CSV names to standardized names for searching
    background_name_map = {
        "Feild Background": "Field Background",
        "white Background": "White Background",
    }

    # Get the standardized name
    search_name = background_name_map.get(background_part, background_part)

    # Dynamically find the actual folder name (handles trailing spaces)
    actual_background_folder = find_matching_folder(BASE_PATH, search_name)

    full_path = os.path.join(BASE_PATH, actual_background_folder, disease_folder, filename)

    return full_path, background_part, disease_folder


# Apply path building
df[["original_path", "original_background", "original_disease"]] = df.apply(
    lambda row: pd.Series(build_original_path(row)), axis=1
)

# Verify file existence
print("\nVerify image files...")
missing_count = 0
for idx, path in enumerate(df["original_path"].head(20)):
    if not os.path.exists(path):
        print(f"  ⚠ File does not exist: {path}")
        missing_count += 1

if missing_count == 0:
    print(f"✓ First 20 file paths verified")
else:
    print(f"⚠ Warning: {missing_count} files not found")
    response = input("\nContinue? (y/n): ")
    if response.lower() != "y":
        print("Canceled")
        exit()

# ===== Step 3: Standardize Labels =====
print("\n[Step 3] Standardize Disease and Background Labels")
print("-" * 60)


def standardize_disease_name(disease_str):
    """Standardize disease name"""
    disease_str = disease_str.replace("Browon Spot", "Brown Spot")
    disease_str = disease_str.replace("Shath Blight", "Sheath Blight")
    disease_str = disease_str.replace("Rice Turgro", "Rice Tungro")
    disease_str = disease_str.replace("Leaf Scaled", "Leaf Scald")
    return disease_str


def standardize_background_name(bg_str):
    """Standardize background name"""
    bg_str = bg_str.replace("Feild", "Field")
    bg_str = bg_str.replace("white", "White")
    return bg_str


df["disease_clean"] = df["original_disease"].apply(standardize_disease_name)
df["background_clean"] = df["original_background"].apply(standardize_background_name)

disease_to_label = {
    "Brown Spot": 0,
    "Leaf Scald": 1,
    "Rice Blast": 2,
    "Rice Tungro": 3,
    "Sheath Blight": 4,
}

background_to_label = {"Field Background": 0, "White Background": 1}

df["disease_label"] = df["disease_clean"].map(disease_to_label)
df["background_label"] = df["background_clean"].map(background_to_label)

unmapped_diseases = df[df["disease_label"].isna()]["disease_clean"].unique()
unmapped_backgrounds = df[df["background_label"].isna()]["background_clean"].unique()

if len(unmapped_diseases) > 0:
    print(f"⚠ Warning: Unmapped diseases: {unmapped_diseases}")
if len(unmapped_backgrounds) > 0:
    print(f"⚠ Warning: Unmapped backgrounds: {unmapped_backgrounds}")

df["is_augmented"] = False
df["augmentation_type"] = "original"
df["original_image"] = df["pictureName"]

print(f"✓ Standardization completed")
print(f"\nStandardized data distribution:")
print(df.groupby(["disease_clean", "background_clean"]).size())

# ===== Step 4: Enhanced Data Augmentation Functions =====
print("\n[Step 4] Prepare Enhanced Data Augmentation Functions")
print("-" * 60)


def rotate_image(image, angle):
    """Rotate image by specified angle"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated


def flip_image(image, flip_code):
    """Flip image"""
    return cv2.flip(image, flip_code)


def scale_image(image, scale_factor):
    """Scale image by specified factor"""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale_factor < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        scaled = cv2.copyMakeBorder(
            scaled,
            pad_h,
            h - new_h - pad_h,
            pad_w,
            w - new_w - pad_w,
            cv2.BORDER_REFLECT,
        )
    else:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        scaled = scaled[start_h : start_h + h, start_w : start_w + w]

    return scaled


def blur_image(image, kernel_size):
    """Apply Gaussian blur with specified kernel size"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_brightness(image, factor):
    """Adjust image brightness"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(image, factor):
    """Adjust image contrast"""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def add_noise(image, noise_level):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def augment_image(image, aug_type, params=None):
    """
    Apply augmentation with random parameters
    Returns: (augmented_image, parameter_description)
    """
    if aug_type == "rotate":
        angle = random.uniform(-30, 30) if params is None else params
        return rotate_image(image, angle), f"angle={angle:.1f}"

    elif aug_type == "flip_h":
        return flip_image(image, 1), "horizontal"

    elif aug_type == "flip_v":
        return flip_image(image, 0), "vertical"

    elif aug_type == "scale":
        scale = random.uniform(0.85, 0.95) if params is None else params
        return scale_image(image, scale), f"scale={scale:.2f}"

    elif aug_type == "blur":
        kernel = random.choice([3, 5, 7]) if params is None else params
        return blur_image(image, kernel), f"kernel={kernel}"

    elif aug_type == "brightness":
        factor = random.uniform(0.7, 1.3) if params is None else params
        return adjust_brightness(image, factor), f"bright={factor:.2f}"

    elif aug_type == "contrast":
        factor = random.uniform(0.8, 1.2) if params is None else params
        return adjust_contrast(image, factor), f"contrast={factor:.2f}"

    elif aug_type == "noise":
        level = random.uniform(5, 15) if params is None else params
        return add_noise(image, level), f"noise={level:.1f}"

    elif aug_type == "combo":
        img = image.copy()
        desc_parts = []

        if random.random() > 0.5:
            angle = random.uniform(-20, 20)
            img = rotate_image(img, angle)
            desc_parts.append(f"rot{angle:.1f}")

        if random.random() > 0.5:
            img = flip_image(img, 1)
            desc_parts.append("flipH")

        if random.random() > 0.4:
            kernel = random.choice([3, 5])
            img = blur_image(img, kernel)
            desc_parts.append(f"blur{kernel}")

        if random.random() > 0.4:
            factor = random.uniform(0.8, 1.2)
            img = adjust_brightness(img, factor)
            desc_parts.append(f"br{factor:.2f}")

        return img, "+".join(desc_parts) if desc_parts else "combo"

    return image, "none"


print("✓ Enhanced augmentation functions prepared")
print("  Available types: rotate, flip_h, flip_v, scale, blur, brightness, contrast, noise, combo")

# ===== Step 5: Execute Enhanced Data Augmentation =====
print("\n[Step 5] Execute Enhanced Data Augmentation")
print("-" * 60)

augmented_base = os.path.join(OUTPUT_PATH)
os.makedirs(augmented_base, exist_ok=True)

# Updated augmentation types (more diverse)
aug_types = ["rotate", "flip_h", "flip_v", "scale", "blur", "brightness", "contrast", "noise", "combo"]
augmented_records = []

groups = df.groupby(["disease_clean", "background_clean"])

for (disease, background), group in groups:
    current_count = len(group)
    needed = TARGET_IMAGES_PER_CLASS - current_count

    print(f"\nProcessing: {disease} ({background})")
    print(f"  Current: {current_count} images")

    if needed <= 0:
        print(f"  ✓ Already met target, skipping")
        continue

    print(f"  Need to generate: {needed} images")

    output_dir = os.path.join(augmented_base, background, disease)
    os.makedirs(output_dir, exist_ok=True)

    # Track used combinations to avoid duplicates
    used_combinations = set()

    aug_count = 0
    pbar = tqdm(total=needed, desc=f"  Augmenting")

    # Strategy: First apply each augmentation type to each image once
    original_images = group["original_path"].tolist()
    original_names = group["pictureName"].tolist()

    # Phase 1: Systematic augmentation (each type applied once to each image)
    phase1_limit = min(needed, len(original_images) * len(aug_types))
    img_idx = 0
    type_idx = 0

    while aug_count < phase1_limit:
        original_path = original_images[img_idx]
        original_name = original_names[img_idx]
        aug_type = aug_types[type_idx]

        # Create unique combination key
        combo_key = (original_name, aug_type)

        if combo_key in used_combinations:
            type_idx = (type_idx + 1) % len(aug_types)
            if type_idx == 0:
                img_idx = (img_idx + 1) % len(original_images)
            continue

        if not os.path.exists(original_path):
            type_idx = (type_idx + 1) % len(aug_types)
            if type_idx == 0:
                img_idx = (img_idx + 1) % len(original_images)
            continue

        image = cv2.imread(original_path)
        if image is None:
            type_idx = (type_idx + 1) % len(aug_types)
            if type_idx == 0:
                img_idx = (img_idx + 1) % len(original_images)
            continue

        augmented_img, param_desc = augment_image(image, aug_type)

        base_name = Path(original_name).stem
        ext = Path(original_name).suffix
        new_filename = f"{base_name}_aug{aug_count:04d}_{aug_type}{ext}"
        output_path = os.path.join(output_dir, new_filename)

        success = cv2.imwrite(output_path, augmented_img)
        if success:
            used_combinations.add(combo_key)

            # Find original row
            original_row = group[group["pictureName"] == original_name].iloc[0]

            augmented_records.append({
                "pictureName": new_filename,
                "Diseases": f"{disease}({background})",
                "original_path": output_path,
                "original_background": background,
                "original_disease": disease,
                "disease_clean": disease,
                "background_clean": background,
                "disease_label": original_row["disease_label"],
                "background_label": original_row["background_label"],
                "is_augmented": True,
                "augmentation_type": f"{aug_type}({param_desc})",
                "original_image": original_name,
            })

            aug_count += 1
            pbar.update(1)

        type_idx = (type_idx + 1) % len(aug_types)
        if type_idx == 0:
            img_idx = (img_idx + 1) % len(original_images)

    # Phase 2: Random augmentation with diversity (if still needed)
    max_attempts = needed * 10  # Prevent infinite loop
    attempts = 0

    while aug_count < needed and attempts < max_attempts:
        attempts += 1

        # Randomly select original image
        sample_idx = random.randint(0, len(original_images) - 1)
        original_path = original_images[sample_idx]
        original_name = original_names[sample_idx]

        # Randomly select augmentation type
        aug_type = random.choice(aug_types)

        # For deterministic augmentations, skip if already used
        if aug_type in ["flip_h", "flip_v"]:
            combo_key = (original_name, aug_type)
            if combo_key in used_combinations:
                continue
            used_combinations.add(combo_key)

        if not os.path.exists(original_path):
            continue

        image = cv2.imread(original_path)
        if image is None:
            continue

        augmented_img, param_desc = augment_image(image, aug_type)

        base_name = Path(original_name).stem
        ext = Path(original_name).suffix
        new_filename = f"{base_name}_aug{aug_count:04d}_{aug_type}{ext}"
        output_path = os.path.join(output_dir, new_filename)

        success = cv2.imwrite(output_path, augmented_img)
        if success:
            original_row = group[group["pictureName"] == original_name].iloc[0]

            augmented_records.append({
                "pictureName": new_filename,
                "Diseases": f"{disease}({background})",
                "original_path": output_path,
                "original_background": background,
                "original_disease": disease,
                "disease_clean": disease,
                "background_clean": background,
                "disease_label": original_row["disease_label"],
                "background_label": original_row["background_label"],
                "is_augmented": True,
                "augmentation_type": f"{aug_type}({param_desc})",
                "original_image": original_name,
            })

            aug_count += 1
            pbar.update(1)

    pbar.close()

    if aug_count < needed:
        print(f"  ⚠ Warning: Only generated {aug_count}/{needed} images (may need more diverse augmentations)")

print(f"\n✓ Augmentation completed! Generated {len(augmented_records)} new images")

# ===== Step 6: Merge Data and Split =====
print("\n[Step 6] Merge Data and Split into Train/Validation/Test")
print("-" * 60)

augmented_df = pd.DataFrame(augmented_records)
final_df = pd.concat([df, augmented_df], ignore_index=True)

final_df["split"] = "train"
train_indices, val_indices, test_indices = [], [], []

groups = final_df.groupby(["disease_clean", "background_clean"])
for (disease, background), group in groups:
    indices = group.index.tolist()

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=RANDOM_SEED
    )

    train_indices.extend(train_idx)
    val_indices.extend(val_idx)
    test_indices.extend(test_idx)

final_df.loc[train_indices, "split"] = "train"
final_df.loc[val_indices, "split"] = "validation"
final_df.loc[test_indices, "split"] = "test"

final_df.to_csv(OUTPUT_CSV, index=False)

print(f"✓ Dataset splitting completed")
print(f"\nDataset statistics:")
print(f"  Total: {len(final_df)} images")
print(f"  Original: {len(df)} images")
print(f"  Augmented: {len(augmented_df)} images")
print(f"\nSplit distribution:")
print(final_df["split"].value_counts())

print(f"\nDisease distribution in each split:")
for split_name in ["train", "validation", "test"]:
    print(f"\n{split_name.upper()}:")
    split_data = final_df[final_df["split"] == split_name]
    print(split_data.groupby("disease_clean").size())

print("\n" + "=" * 60)
print(f"✓ All completed!")
print(f"✓ New CSV saved: {OUTPUT_CSV}")
print(f"✓ Augmented images: {augmented_base}")
print("=" * 60)
