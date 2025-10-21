import os
import glob
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

# --- 1. 配置参数 ---
# 您的源数据文件夹，应包含所有类别的子文件夹
SOURCE_DATA_DIR = '../Dhan-Shomadhan'
TEMP_DATA_DIR = 'kfold_temp_data' # 每次折叠时临时存放数据的目录
N_SPLITS = 5                     # K-Fold 的 K 值
PROJECT_NAME = 'yolo_cls_kfold'  # 保存训练结果的项目名

# --- 从您的配置中应用的参数 (2025-10-21 更新) ---
MODEL_NAME = 'yolo11s-cls.pt' # 请确保此模型文件存在
EPOCHS = 150
IMGSZ = 320
BATCH_SIZE = 16
DEVICE = 0
FREEZE = 0
PATIENCE = 30
OPTIMIZER = 'AdamW'
LR0 = 0.0005
AUGMENT = True
CACHE = True
WORKERS = 8
# --- 数据增强参数 ---
DEGREES = 10
TRANSLATE = 0.05
SCALE = 0.05
FLIPLR = 0.3
# --- 结束应用 ---


# --- 2. 收集所有图片和标签 ---
print("Collecting image paths and labels...")
image_paths = []
labels = []
class_names = sorted([d.name for d in os.scandir(SOURCE_DATA_DIR) if d.is_dir()])
class_to_idx = {name: i for i, name in enumerate(class_names)}

if not class_names:
    print(f"Error: No class subdirectories found in {SOURCE_DATA_DIR}.")
    print("Please ensure your data directory is structured like: 'dataset_yolo/class_a/...'")
    print("If your class names have spaces, please rename them (e.g., 'Brown Spot' -> 'Brown_Spot')")
    exit()

for class_name in class_names:
    if ' ' in class_name:
        print(f"Warning: Class name '{class_name}' contains a space.")
        print("This may cause read errors (WARN:0... can't open/read file).")
        print("It is HIGHLY recommended to rename this directory to replace spaces (e.g., 'Brown_Spot').")

    class_dir = os.path.join(SOURCE_DATA_DIR, class_name)
    # 查找多种常见图片格式
    files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG'):
        files.extend(glob.glob(os.path.join(class_dir, ext)))
        
    image_paths.extend(files)
    labels.extend([class_to_idx[class_name]] * len(files))

image_paths = np.array(image_paths)
labels = np.array(labels)

if len(image_paths) == 0:
    print(f"Error: No images found in {SOURCE_DATA_DIR}. Searched for .jpg, .jpeg, .png, etc.")
    exit()

print(f"Found {len(image_paths)} images belonging to {len(class_names)} classes.")

# --- 3. 初始化 StratifiedKFold ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_results = [] # 存储每一折的最佳 top1 准确率

# --- 4. K-Fold 循环 ---
for fold, (train_indices, val_indices) in enumerate(skf.split(image_paths, labels)):
    fold_num = fold + 1
    print(f"\n--- Starting Fold {fold_num}/{N_SPLITS} ---")

    # --- 4a. 清理并创建临时的 train/val 目录结构 ---
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
    
    for split in ['train', 'val']:
        for class_name in class_names:
            os.makedirs(os.path.join(TEMP_DATA_DIR, split, class_name), exist_ok=True)

    # --- 4b. 复制文件到临时目录 ---
    print(f"Copying files for fold {fold_num}...")
    
    # 复制训练集
    for idx in train_indices:
        src_path = image_paths[idx]
        label_name = class_names[labels[idx]]
        dst_path = os.path.join(TEMP_DATA_DIR, 'train', label_name, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
    # 复制验证集
    for idx in val_indices:
        src_path = image_paths[idx]
        label_name = class_names[labels[idx]]
        dst_path = os.path.join(TEMP_DATA_DIR, 'val', label_name, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        
    print("File copying complete.")

    # --- 4c. 训练模型 ---
    # 每次都从同一个预训练模型加载
    model = YOLO(MODEL_NAME) 

    run_name = f'fold_{fold_num}'
    
    print(f"Starting training for fold {fold_num} with specified parameters...")
    
    try:
        results = model.train(
            data=TEMP_DATA_DIR, # 数据路径指向我们刚创建的临时目录
            
            # --- 应用您的所有配置参数 ---
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device=DEVICE,
            freeze=FREEZE,
            patience=PATIENCE,
            optimizer=OPTIMIZER,
            lr0=LR0,
            augment=AUGMENT,
            cache=CACHE,
            workers=WORKERS,
            # --- 数据增强参数 ---
            degrees=DEGREES,
            translate=TRANSLATE,
            scale=SCALE,
            fliplr=FLIPLR,
            # --- 结束应用 ---

            # K-Fold 脚本控制的参数
            project=PROJECT_NAME,
            name=run_name,
            exist_ok=True # 如果同名运行存在，则覆盖
        )
        
        # --- 4d. 收集结果 ---
        best_top1_accuracy = results.top1
        fold_results.append(best_top1_accuracy)
        
        print(f"--- Fold {fold_num} Finished ---")
        print(f"Best Top-1 Accuracy: {best_top1_accuracy:.4f}")

    except Exception as e:
        print(f"Error during training for fold {fold_num}: {e}")
        print("Skipping this fold.")


# --- 5. 清理最后的临时数据 ---
if os.path.exists(TEMP_DATA_DIR):
    shutil.rmtree(TEMP_DATA_DIR)
    print(f"Cleaned up temporary data directory: {TEMP_DATA_DIR}")

# --- 6. 聚合与评估结果 ---
if fold_results:
    mean_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)
    
    print("\n--- K-Fold Cross-Validation Final Results ---")
    print(f"Metrics collected (Best Top-1 Accuracy): {fold_results}")
    print(f"Average Top-1 Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: {std_accuracy:.4f}")
else:
    print("No results were collected. Please check for errors during training.")
