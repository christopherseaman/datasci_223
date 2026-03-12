# Assignment 5: Classification Showdown

Build and compare classification models on the **Fashion-MNIST** dataset. You'll work through three classification tasks of increasing complexity, practicing the full model comparison workflow from lecture.

## Dataset

Fashion-MNIST contains 70,000 grayscale images (28×28 pixels) of clothing items across 10 categories:

| Label | Category |
|:---:|:---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Learning Objectives

- Filter and prepare data for different classification tasks
- Apply train/test splitting with stratification
- Use StandardScaler properly (fit on train only!)
- Train and compare multiple models using cross-validation
- Select the best model and evaluate on held-out test data
- Generate evaluation artifacts (metrics, plots)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Your Tasks

Complete the code cells in `assignment.md` (or `assignment.ipynb` after converting). The notebook is organized into three parts that build on each other.

---

### Part 1: Binary Classification

**Task:** Classify T-shirts (label 0) vs Trousers (label 1)

**What to implement in `part1_binary_classification()`:**

1. Use `load_fashion_mnist()` from helpers to get the full dataset
2. Filter to only include samples with labels 0 and 1
3. Split into train/test sets (80/20) with stratification
4. Scale features using StandardScaler (fit on train, transform both)
5. Train a LogisticRegression model
6. Evaluate: calculate accuracy and confusion matrix
7. Save results to `output/part1_results.json`

**Output format (`output/part1_results.json`):**
```json
{
  "task": "tshirt_vs_trouser",
  "accuracy": 0.95,
  "confusion_matrix": [[TN, FP], [FN, TP]]
}
```

---

### Part 2: Multi-class Classification with Cross-Validation

**Task:** Classify footwear types: Sandal (5) vs Sneaker (7) vs Ankle boot (9)

**What to implement in `part2_multiclass_cv()`:**

1. Filter dataset to labels 5, 7, and 9
2. Split into train/test sets (80/20) with stratification
3. Scale features
4. Encode labels to 0, 1, 2 using LabelEncoder (required for XGBoost)
5. Compare three models using 5-fold cross-validation:
   - LogisticRegression
   - RandomForestClassifier
   - XGBClassifier
6. Select the best model based on mean CV accuracy
7. Retrain the best model on the full training set
8. Evaluate on test set using `classification_report`
9. Save CV results and final evaluation

**Output files:**
- `output/part2_cv_results.csv` - columns: `model`, `fold`, `score`
- `output/part2_test_results.csv` - per-class precision, recall, F1 from classification_report

---

### Part 3: Full Model Comparison Pipeline

**Task:** Classify Clothing vs Footwear (binary)

- **Clothing (label 0):** T-shirt, Trouser, Pullover, Dress, Coat, Shirt (labels 0-4, 6)
- **Footwear (label 1):** Sandal, Sneaker, Ankle boot (labels 5, 7, 9)
- **Exclude:** Bag (label 8)

**What to implement in `part3_full_pipeline()`:**

1. Create binary labels from the original 10-class labels (exclude bags)
2. Split into train/test sets (80/20) with stratification
3. Scale features
4. Define three models:
   - LogisticRegression
   - RandomForestClassifier
   - XGBClassifier
5. Run 5-fold cross-validation on ALL models using the **same** StratifiedKFold splitter
6. Select the best model based on mean CV AUC score
7. Retrain the best model on the full training set
8. Generate ROC curves for all models (use provided helper)
9. Generate confusion matrix for the best model (use provided helper)
10. Save all results

**Output files:**
- `output/part3_cv_results.csv` - columns: `model`, `fold`, `auc`
- `output/part3_test_results.csv` - columns: `model`, `accuracy`, `auc` (best model only)
- `output/part3_roc_curves.png` - ROC curves for all 3 models
- `output/part3_confusion_matrix.png` - confusion matrix for best model

---

## Running Your Code

Convert the markdown to a Jupyter notebook and run it:

```bash
jupytext --to notebook assignment.md
jupyter nbconvert --execute assignment.ipynb
```

Or open `assignment.ipynb` in Jupyter/VS Code and run the cells interactively.

## Checking Your Work

Run the autograder tests locally:

```bash
pytest .github/tests/ -v
```

## Output Files Summary

After successful completion, your `output/` directory should contain:

| File | Part | Description |
|:---|:---:|:---|
| `part1_results.json` | 1 | Accuracy and confusion matrix |
| `part2_cv_results.csv` | 2 | CV scores for all three models |
| `part2_test_results.csv` | 2 | Per-class metrics |
| `part3_cv_results.csv` | 3 | CV AUC scores for all models |
| `part3_test_results.csv` | 3 | Final test metrics |
| `part3_roc_curves.png` | 3 | ROC curve comparison |
| `part3_confusion_matrix.png` | 3 | Best model confusion matrix |

## Hints

See `hints.md` if you get stuck.
