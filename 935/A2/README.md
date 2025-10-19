# Rice Disease Classification System

A deep learning-based system for classifying rice leaf diseases using YOLOv8s classification model.

## Project Structure

```
A2/
├── main.py                  # Main entry point with CLI
├── config.py               # Configuration settings
├── data_preprocessor.py    # Dataset preparation and splitting
├── trainer.py              # Model training module
├── evaluator.py            # Model evaluation module
├── inference.py            # Inference on single/batch images
├── feature_visualizer.py   # Feature visualization module
├── requirements.txt        # Python dependencies
├── Dhan-Shomadhan/        # Original dataset (not included in submission)
├── dataset/               # Processed dataset (generated)
├── models/                # Trained models (generated)
└── results/               # Training/evaluation results (generated)
```

## Installation

### Prerequisites
- Python 3.12
- CUDA-capable GPU (optional, for faster training)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Verify Dataset Structure (Optional)

Check if dataset is in correct location:

```bash
python setup_dataset.py --verify
```

### 2. Prepare Dataset

Split the dataset into train/validation/test sets (70%/15%/15%):

```bash
# Simple (uses default location: ./Dhan-Shomadhan/processed_data/)
python main.py prepare

# Or explicitly specify paths
python main.py prepare --source ./Dhan-Shomadhan --seed 42
```

**Note:** Processed data is automatically created under `./Dhan-Shomadhan/processed_data/` as per assignment requirements.

### 3. Train Model

Train the YOLOv8s classification model:

```bash
# Using CUDA (recommended)
python main.py train --epochs 100 --batch-size 16 --device cuda

# Using CPU
python main.py train --epochs 100 --batch-size 8 --device cpu

# Resume training from checkpoint
python main.py train --resume
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--imgsz`: Input image size (default: 224)
- `--device`: Device to use - cuda/cpu (default: auto-detect)

### 4. Evaluate Model

Evaluate the trained model on different scenarios:

```bash
# Evaluate on test set
python main.py evaluate --model ./models/best_model.pt --split test

# Evaluate on validation set
python main.py evaluate --model ./models/best_model.pt --split val

# Evaluate on three scenarios (white background, field background, mixed)
python main.py evaluate --model ./models/best_model.pt --scenarios
```

### 5. Run Inference

Predict disease class for new images:

```bash
# Single image inference (CPU)
python main.py infer --model ./models/best_model.pt --source ./test_image.jpg --device cpu

# Single image inference (CUDA)
python main.py infer --model ./models/best_model.pt --source ./test_image.jpg --device cuda

# Batch inference on folder
python main.py infer --model ./models/best_model.pt --source ./test_folder/ --device cuda

# Inference without saving results
python main.py infer --model ./models/best_model.pt --source ./image.jpg --no-save
```

### 6. Visualize Learned Features

Demonstrate that the model learns disease features, not background:

```bash
# Visualize features for a single image
python main.py visualize --model ./models/best_model.pt --image ./test_image.jpg

# Compare predictions across different backgrounds
python main.py visualize --model ./models/best_model.pt \
    --white-bg ./white_background_sample.jpg \
    --field-bg ./field_background_sample.jpg
```

### 7. Run K-Fold Cross-Validation

Perform rigorous 5-fold cross-validation:

```bash
# Run 5-fold cross-validation
python main.py cross-validate --n-folds 5 --epochs 100 --device cuda

# Run with different number of folds
python main.py cross-validate --n-folds 10 --epochs 50 --device cuda
```

**What is K-Fold Cross-Validation?**
- Splits data into K folds (e.g., 5 folds)
- Trains K models, each using different fold as test set
- Reports mean ± std across all folds
- More robust than single train/test split
- Ensures results generalize across different data splits

## Disease Classes

The system classifies five rice diseases:

1. **Brown Spot** - Fungal disease causing brown lesions
2. **Leaf Scald** - Bacterial disease with scalded appearance
3. **Rice Blast** - Fungal disease causing blast-like lesions
4. **Rice Tungro** - Viral disease causing yellow-orange discoloration
5. **Sheath Blight** - Fungal disease affecting leaf sheaths

## Model Architecture

- **Base Model**: YOLOv8s-cls (classification variant)
- **Input Size**: 224×224 pixels
- **Classes**: 5 disease types
- **Training Strategy**: Transfer learning with pre-trained weights

## Output Files

### Training Outputs
- `models/best_model.pt` - Best model checkpoint
- `results/train_*/weights/` - Model weights
- `results/train_*/` - Training metrics, loss curves, confusion matrices

### Evaluation Outputs
- `results/eval_*/confusion_matrix_*.png` - Confusion matrices
- `results/eval_*/metrics_*.json` - Detailed metrics
- `results/eval_*/scenario_comparison.png` - Performance across scenarios

### Inference Outputs
- `results/predictions/pred_*.png` - Visualization of predictions
- `results/predictions/predictions.txt` - Prediction results
- `results/predictions/class_distribution.png` - Distribution plot

### Feature Visualization Outputs
- `results/feature_visualization/feature_viz_*.png` - Feature analysis
- `results/background_comparison/background_comparison.png` - Background independence analysis

## Reproducibility

All random operations use a fixed seed (default: 42) for reproducibility:
- Dataset splitting
- Model initialization
- Training process

To use a different seed, specify it during dataset preparation:
```bash
python main.py prepare --seed 123
```

## Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 8GB
- Storage: 5GB

### Recommended (for training)
- GPU: NVIDIA GPU with 6GB+ VRAM
- RAM: 16GB
- Storage: 10GB

## Notes

- The model is trained on both white background and field background images
- A single unified model handles all three evaluation scenarios
- CUDA is optional but strongly recommended for training
- Inference can run efficiently on CPU
