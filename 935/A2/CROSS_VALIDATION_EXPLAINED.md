# Cross-Validation vs Multiple Independent Runs

## Two Approaches Available

This project provides **TWO** different statistical validation approaches:

---

## Approach 1: K-Fold Cross-Validation (NEW!)

### What is it?
K-Fold Cross-Validation splits the **entire dataset** into K equal folds, then trains K models where each fold serves as the test set once.

### How it works (5-fold example):
```
Dataset (1106 images)
├── Fold 1: Test (221 images)  │ Train+Val: 885 images
├── Fold 2: Test (221 images)  │ Train+Val: 885 images
├── Fold 3: Test (221 images)  │ Train+Val: 885 images
├── Fold 4: Test (221 images)  │ Train+Val: 885 images
└── Fold 5: Test (222 images)  │ Train+Val: 884 images
```

- **Fold 1**: Train on folds 2-5, test on fold 1
- **Fold 2**: Train on folds 1,3-5, test on fold 2
- **Fold 3**: Train on folds 1-2,4-5, test on fold 3
- **Fold 4**: Train on folds 1-3,5, test on fold 4
- **Fold 5**: Train on folds 1-4, test on fold 5

### Usage:
```bash
python main.py cross-validate --n-folds 5 --epochs 100 --device cuda
```

### Advantages:
✅ Every image is used for testing exactly once
✅ Every image is used for training K-1 times
✅ Maximum utilization of limited data
✅ More statistically rigorous
✅ Better for small datasets
✅ Standard practice in machine learning research

### Outputs:
- `results/cross_validation_*/fold_X/` - Results for each fold
- `results/cross_validation_*/cv_statistics.json` - Mean ± Std statistics
- `results/cross_validation_*/cv_results.png` - Per-fold performance chart
- `results/cross_validation_*/cv_summary.png` - Overall summary chart
- `results/cross_validation_*/cv_results_table.tex` - LaTeX table

### When to use:
- **Academic papers and research** (gold standard)
- When you want maximum statistical rigor
- When dataset is limited
- When you want to prove generalization

---

## Approach 2: Multiple Independent Runs

### What is it?
Creates **5 completely different random splits** of the data, trains 5 independent models.

### How it works (5 runs example):
```
Run 1: 70% train | 15% val | 15% test  (seed=42)
Run 2: 70% train | 15% val | 15% test  (seed=43)
Run 3: 70% train | 15% val | 15% test  (seed=44)
Run 4: 70% train | 15% val | 15% test  (seed=45)
Run 5: 70% train | 15% val | 15% test  (seed=46)
```

Each run has **completely different** train/val/test splits.

### Usage:
```bash
python run_experiments.py --n-runs 5 --epochs 100 --device cuda
```

### Advantages:
✅ Tests robustness to different data splits
✅ Simpler to understand
✅ Each run is completely independent
✅ Good for sensitivity analysis

### Outputs:
- `results/experiments_*/run_X/` - Results for each run
- `results/experiments_*/statistics.json` - Mean ± Std statistics
- `results/experiments_*/overall_statistics.png` - Summary chart
- `results/experiments_*/results_table.tex` - LaTeX table

### When to use:
- When you want to test sensitivity to data splitting
- When you have enough data
- When you want independent experiments

---

## Key Differences

| Aspect | K-Fold Cross-Validation | Multiple Independent Runs |
|--------|------------------------|---------------------------|
| **Data usage** | Each sample used in test exactly once | Each sample may/may not be in test |
| **Training data** | Maximized (K-1)/K of data | Fixed 70% of data |
| **Statistical rigor** | Higher (standard in ML) | Lower |
| **Independence** | Folds are not independent | Runs are completely independent |
| **Computation** | K models trained | N models trained |
| **Best for** | Small datasets, research | Sensitivity analysis |
| **Recommended by** | Academic community | Industry practice |

---

## Which One Should You Use?

### For Academic Reports (Recommended):
```bash
# Use K-Fold Cross-Validation
python main.py cross-validate --n-folds 5 --epochs 100 --device cuda
```

**Why?**
- ✅ More rigorous statistically
- ✅ Standard in machine learning papers
- ✅ Better use of limited data (1106 images)
- ✅ More convincing results
- ✅ Higher academic credibility

### For Additional Analysis (Optional):
```bash
# Also run independent experiments for comparison
python run_experiments.py --n-runs 5 --epochs 100 --device cuda
```

**Why?**
- ✅ Shows consistency across different splits
- ✅ Additional evidence of robustness
- ✅ Can report both in your report

---

## Example Results Reporting

### In Your Report - Method Section:

**If using Cross-Validation:**
> "We employ 5-fold stratified cross-validation to ensure robust performance estimation. The dataset is split into 5 folds, maintaining class distribution. Each fold serves as the test set once while the remaining 4 folds are used for training (85%) and validation (15%). We report the mean accuracy and standard deviation across all 5 folds."

**If using Independent Runs:**
> "To assess model stability, we train the model 5 times with different random seeds (42-46). Each run uses a 70/15/15 split for train/validation/test. We report the mean accuracy and standard deviation across all 5 runs."

### In Your Report - Results Section:

**Cross-Validation Results Table:**
```
Scenario              Mean ± Std       Min      Max
─────────────────────────────────────────────────
Test Set             0.9245 ± 0.0134  0.9102   0.9387
White Background     0.9534 ± 0.0089  0.9421   0.9645
Field Background     0.8512 ± 0.0201  0.8245   0.8756
Mixed                0.9156 ± 0.0156  0.8934   0.9345
```

---

## Assignment Requirement Interpretation

The assignment states:
> "All results should be an average over 5 runs with different random splits (same percentages) of training, validation (if applicable), and testing."

This can be interpreted as:

1. **5-fold cross-validation** ✅ (5 different test sets)
2. **5 independent runs** ✅ (5 different random splits)

**Both are valid!** Cross-validation is generally preferred in academic contexts.

---

## Computational Cost

### 5-Fold Cross-Validation:
- **Training time**: ~5 × training time per fold
- **GPU memory**: Same as single run
- **Storage**: 5 models + results
- **Example**: 30 min/fold × 5 = 2.5 hours (GPU)

### 5 Independent Runs:
- **Training time**: ~5 × training time per run
- **GPU memory**: Same as single run
- **Storage**: 5 models + results
- **Example**: 30 min/run × 5 = 2.5 hours (GPU)

**Both take approximately the same time!**

---

## Recommendation for Your Assignment

### Best Practice:
```bash
# 1. Run 5-fold cross-validation (primary results)
python main.py cross-validate --n-folds 5 --epochs 100 --device cuda

# 2. Also evaluate on the three scenarios
# (This is automatically done in cross-validation)
```

### Report in your paper:
1. **Primary results**: 5-fold cross-validation
2. **Scenario analysis**: White background vs Field background vs Mixed
3. **Statistical significance**: Mean ± Std for all scenarios

This gives you:
- ✅ Rigorous statistical validation
- ✅ All three scenarios evaluated
- ✅ Comprehensive results for your report
- ✅ Academic credibility

---

## Quick Reference

```bash
# K-Fold Cross-Validation (RECOMMENDED)
python main.py cross-validate --n-folds 5 --epochs 100 --device cuda

# Multiple Independent Runs (ALTERNATIVE)
python run_experiments.py --n-runs 5 --epochs 100 --device cuda

# Single train/test (for quick testing)
python main.py prepare
python main.py train --epochs 100 --device cuda
python main.py evaluate --model ./models/best_model.pt --scenarios
```

---

## Summary

**You now have THREE options:**

1. ✅ **K-Fold Cross-Validation** (cross_validator.py) - RECOMMENDED FOR REPORT
2. ✅ **Multiple Independent Runs** (run_experiments.py) - ALTERNATIVE
3. ✅ **Single Train/Test Split** (main.py) - QUICK TESTING

Choose based on your needs. For academic rigor, use option 1!
