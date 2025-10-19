# User Manual - Rice Disease Classification System

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start Guide](#quick-start-guide)
4. [Detailed Usage](#detailed-usage)
5. [Reproducing Report Results](#reproducing-report-results)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- Python 3.12
- 8GB RAM
- 5GB free disk space
- CPU with 4+ cores

### Recommended for Training
- NVIDIA GPU with 6GB+ VRAM (CUDA support)
- 16GB RAM
- 10GB free disk space

## Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch>=2.0.0` - PyTorch deep learning framework
- `ultralytics>=8.0.0` - YOLOv8 implementation
- `opencv-python>=4.8.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Plotting and visualization
- `seaborn>=0.12.0` - Statistical visualization
- `scikit-learn>=1.3.0` - Machine learning utilities
- `Pillow>=10.0.0` - Image handling
- `PyYAML>=6.0` - YAML configuration

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

### Step 3: Download Pre-trained Model (Automatic)

The YOLOv8s-cls pre-trained model will be downloaded automatically during first training. No manual download is required.

## Quick Start Guide

### Complete Workflow

```bash
# 1. Prepare dataset
python main.py prepare

# 2. Train model (with CUDA)
python main.py train --epochs 100 --batch-size 16 --device cuda

# 3. Evaluate on three scenarios
python main.py evaluate --model ./models/best_model.pt --scenarios

# 4. Run inference on test images
python main.py infer --model ./models/best_model.pt --source ./dataset/test/Brown_Spot/

# 5. Visualize learned features
python main.py visualize --model ./models/best_model.pt --image ./dataset/test/Brown_Spot/bs_wb_0.jpg
```

## Detailed Usage

### 1. Dataset Preparation

The dataset must be located at `./Dhan-Shomadhan/` with the following structure:

```
Dhan-Shomadhan/
├── Field Background/
│   ├── Browon Spot/
│   ├── Leaf Scaled/
│   ├── Rice Blast/
│   ├── Rice Turgro/
│   └── Sheath Blight/
└── White Background/
    ├── Brown Spot/
    ├── Leaf Scaled/
    ├── Rice Blast/
    ├── Rice Tungro/
    └── Shath Blight/
```

**Prepare and split dataset:**

```bash
python main.py prepare --source ./Dhan-Shomadhan --output ./dataset --seed 42
```

**Output:**
- `./dataset/train/` - Training set (70%)
- `./dataset/val/` - Validation set (15%)
- `./dataset/test/` - Test set (15%)
- `./dataset/data.yaml` - YOLO configuration file
- `./dataset/dataset_info.json` - Dataset statistics

### 2. Model Training

**Basic training:**

```bash
python main.py train --epochs 100 --batch-size 16 --device cuda
```

**Training parameters:**

- `--epochs` (default: 100): Number of training epochs
- `--batch-size` (default: 16): Batch size for training
  - Use 8 for limited GPU memory
  - Use 32 for high-end GPUs
- `--imgsz` (default: 224): Input image size
- `--device`: Training device
  - `cuda` - Use GPU (recommended)
  - `cpu` - Use CPU (slower)
  - Auto-detect if not specified
- `--resume`: Resume from last checkpoint

**Training outputs:**

- `./models/best_model.pt` - Best model (saved automatically)
- `./results/train_*/weights/best.pt` - Best checkpoint
- `./results/train_*/weights/last.pt` - Last checkpoint
- `./results/train_*/results.csv` - Training metrics
- `./results/train_*/confusion_matrix.png` - Confusion matrix
- `./results/train_*/results.png` - Loss and accuracy curves

**Resume training:**

```bash
python main.py train --resume
```

### 3. Model Evaluation

**Evaluate on test set:**

```bash
python main.py evaluate --model ./models/best_model.pt --split test
```

**Evaluate on validation set:**

```bash
python main.py evaluate --model ./models/best_model.pt --split val
```

**Evaluate on three scenarios (IMPORTANT for report):**

```bash
python main.py evaluate --model ./models/best_model.pt --scenarios
```

This evaluates on:
- **Scenario a)** White background test images
- **Scenario b)** Field background test images
- **Scenario c)** Mixed (all test images)

**Evaluation outputs:**

- `./results/eval_*/confusion_matrix_*.png` - Confusion matrices
- `./results/eval_*/metrics_*.json` - Detailed metrics
- `./results/eval_*/scenario_comparison.png` - Performance comparison chart

### 4. Inference

**Single image prediction (CPU):**

```bash
python main.py infer --model ./models/best_model.pt --source ./image.jpg --device cpu
```

**Single image prediction (GPU):**

```bash
python main.py infer --model ./models/best_model.pt --source ./image.jpg --device cuda
```

**Batch prediction on folder:**

```bash
python main.py infer --model ./models/best_model.pt --source ./test_images/ --device cuda
```

**Inference without saving:**

```bash
python main.py infer --model ./models/best_model.pt --source ./image.jpg --no-save
```

**Inference outputs:**

- `./results/predictions/pred_*.png` - Visualization with top-5 predictions
- `./results/predictions/predictions.txt` - Text file with all predictions
- `./results/predictions/class_distribution.png` - Distribution chart

### 5. Feature Visualization

**Visualize learned features for single image:**

```bash
python main.py visualize --model ./models/best_model.pt --image ./test_image.jpg
```

**Compare white vs field background:**

```bash
python main.py visualize --model ./models/best_model.pt \
    --white-bg ./dataset/test/Brown_Spot/bs_wb_0.jpg \
    --field-bg ./dataset/test/Brown_Spot/bsf_0.jpg
```

**Visualization outputs:**

- `./results/feature_visualization/feature_viz_*.png` - Feature analysis showing:
  - Original image
  - Prediction
  - Edge detection (disease boundaries)
  - Attention heatmap
  - Disease region overlay
  - Color-based analysis
- `./results/background_comparison/background_comparison.png` - Side-by-side comparison

## Reproducing Report Results

### Option 1: Single Run

```bash
# 1. Prepare dataset
python main.py prepare

# 2. Train model
python main.py train --epochs 100 --batch-size 16 --device cuda

# 3. Evaluate on scenarios
python main.py evaluate --model ./models/best_model.pt --scenarios

# 4. Generate feature visualizations
python main.py visualize --model ./models/best_model.pt \
    --white-bg ./dataset/test/Brown_Spot/bs_wb_0.jpg \
    --field-bg ./dataset/test/Brown_Spot/bsf_0.jpg
```

### Option 2: Five Independent Runs (as required by assignment)

```bash
python run_experiments.py --n-runs 5 --epochs 100 --batch-size 16 --device cuda
```

**This will:**
- Run 5 independent experiments with different random splits
- Train a model for each run
- Evaluate each model on all three scenarios
- Compute mean and standard deviation
- Generate statistical plots and LaTeX tables

**Outputs:**
- `./results/experiments_*/statistics.json` - Mean ± Std for all scenarios
- `./results/experiments_*/overall_statistics.png` - Bar chart with error bars
- `./results/experiments_*/results_table.tex` - LaTeX table for report

## Troubleshooting

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
python main.py train --batch-size 8 --device cuda
```

### Model Not Found

**Problem:** `Model not found at ./models/best_model.pt`

**Solution:**
```bash
# Check if training completed successfully
ls -la ./results/train_*/weights/

# Copy manually if needed
cp ./results/train_*/weights/best.pt ./models/best_model.pt
```

### Dataset Not Found

**Problem:** `Processed dataset not found at ./dataset`

**Solution:**
```bash
# Prepare dataset first
python main.py prepare
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Slow Training on CPU

**Problem:** Training is very slow

**Solution:**
- Use GPU if available
- Reduce epochs for testing: `--epochs 10`
- Reduce batch size: `--batch-size 8`

## Performance Expectations

### Training Time
- **GPU (NVIDIA RTX 3060):** ~30-40 minutes for 100 epochs
- **CPU (8 cores):** ~3-4 hours for 100 epochs

### Expected Accuracy
- **Overall:** 85-95%
- **White background:** 90-98%
- **Field background:** 75-90%
- **Mixed:** 85-95%

Note: Actual performance depends on data split and training conditions.

## File Locations

All outputs are organized as follows:

```
A2/
├── dataset/              # Processed dataset
├── models/              # Trained models
│   └── best_model.pt   # Best model (copy of best checkpoint)
└── results/            # All results
    ├── train_*/        # Training outputs
    ├── eval_*/         # Evaluation outputs
    ├── predictions/    # Inference outputs
    ├── feature_visualization/  # Feature analysis
    ├── background_comparison/  # Background comparison
    └── experiments_*/  # Multi-run experiment results
```

## Command Reference

| Command | Purpose |
|---------|---------|
| `python main.py prepare` | Prepare and split dataset |
| `python main.py train` | Train model |
| `python main.py evaluate` | Evaluate model |
| `python main.py infer` | Run inference |
| `python main.py visualize` | Visualize features |
| `python run_experiments.py` | Run multiple experiments |

## Getting Help

```bash
# Main help
python main.py --help

# Command-specific help
python main.py train --help
python main.py evaluate --help
python main.py infer --help
python main.py visualize --help
```

## Notes

- All random operations use seed 42 by default for reproducibility
- The model automatically handles both white and field backgrounds
- CUDA is optional but strongly recommended for training
- Inference can run efficiently on CPU
- Pre-trained YOLOv8s-cls weights are downloaded automatically
