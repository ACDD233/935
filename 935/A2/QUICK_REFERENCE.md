# Quick Reference Card

## Essential Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python test_setup.py

# Check submission
python check_submission.py
```

### Complete Workflow
```bash
# 1. Prepare dataset (70/15/15 split)
python main.py prepare

# 2. Train model
python main.py train --epochs 100 --batch-size 16 --device cuda

# 3. Evaluate on all scenarios
python main.py evaluate --model ./models/best_model.pt --scenarios

# 4. Inference
python main.py infer --model ./models/best_model.pt --source ./test_image.jpg --device cpu

# 5. Visualize features
python main.py visualize --model ./models/best_model.pt --image ./test_image.jpg
```

### For Report (5 Independent Runs)
```bash
python run_experiments.py --n-runs 5 --epochs 100 --device cuda
```

## File Organization

### Core Modules
- `main.py` - Main CLI entry point
- `config.py` - Configuration
- `data_preprocessor.py` - Dataset splitting
- `trainer.py` - Model training
- `evaluator.py` - Model evaluation
- `inference.py` - Prediction
- `feature_visualizer.py` - Feature analysis

### Utilities
- `run_experiments.py` - 5-run experiments
- `test_setup.py` - Setup verification
- `quick_test.py` - Quick functionality test
- `check_submission.py` - Submission checker

### Documentation
- `README.md` - Quick start
- `USER_MANUAL.md` - Complete manual
- `PROJECT_SUMMARY.md` - Project overview
- `requirements.txt` - Dependencies

## Key Parameters

### Training
- `--epochs 100` - Training epochs
- `--batch-size 16` - Batch size (reduce if GPU memory issues)
- `--device cuda` - Use GPU (or `cpu`)
- `--imgsz 224` - Image size

### Evaluation
- `--split test` - Evaluate on test/val/train
- `--scenarios` - Evaluate three scenarios (white/field/mixed)

### Inference
- `--device cpu` - Use CPU (or `cuda`)
- `--source path` - Image file or folder
- `--no-save` - Don't save results

## Expected Outputs

### Training
- `models/best_model.pt` - Best model
- `results/train_*/` - Metrics, plots, checkpoints

### Evaluation
- Confusion matrices for each scenario
- Scenario comparison chart
- Detailed metrics JSON

### Inference
- Prediction visualizations
- Class distribution chart
- Prediction text file

### Feature Visualization
- Attention heatmaps
- Background comparison
- Disease region analysis

## Troubleshooting

### Module Import Errors
```bash
# Verify all imports work
python verify_imports.py

# If any package is missing
pip install -r requirements.txt
```

### CUDA Out of Memory
```bash
python main.py train --batch-size 8 --device cuda
```

### Slow CPU Training
```bash
# Reduce epochs for testing
python main.py train --epochs 10 --device cpu
```

### Missing Dataset
```bash
# Ensure Dhan-Shomadhan folder exists
ls -la ./Dhan-Shomadhan
```

## Performance Targets

- Overall Accuracy: 85-95%
- White Background: 90-98%
- Field Background: 75-90%
- Mixed: 85-95%

## Disease Classes

1. Brown_Spot
2. Leaf_Scald
3. Rice_Blast
4. Rice_Tungro
5. Sheath_Blight

## Important Notes

- Single model handles all scenarios
- Results averaged over 5 runs (use run_experiments.py)
- CUDA optional but recommended for training
- Inference works well on CPU
- All operations use seed 42 for reproducibility

## Submission Package

Include:
- All `.py` files
- All `.txt` files
- All `.md` files

Exclude:
- `dataset/` folder
- `models/` folder
- `results/` folder
- `__pycache__/` folder
- `Dhan-Shomadhan/` folder (not submitted with code)

## Quick Test

```bash
# Quick end-to-end test (10 epochs only)
python quick_test.py
```

## Getting Help

```bash
# General help
python main.py --help

# Command-specific help
python main.py train --help
python main.py evaluate --help
python main.py infer --help
python main.py visualize --help
```
