# Assignment 9: Medical Image Classifier

Build a chest X-ray classifier using PyTorch transfer learning.

## Dataset

CIFAR-10 (used as a stand-in for quick iteration — the techniques transfer directly to medical imaging). Binary classification using 2 classes.

| Label | Category |
|:---:|:---|
| 0 | airplane |
| 1 | automobile |

In practice, this would be "normal" vs "abnormal" chest X-rays, with the same pipeline.

## Learning Objectives

- Load and preprocess images with `torchvision.transforms`
- Build DataLoaders for train/validation/test splits
- Apply transfer learning with a pretrained torchvision model
- Write a PyTorch training loop
- Evaluate with confusion matrix and classification report
- Save and load model weights

## Setup

```bash
pip install -r requirements.txt
```

## Your Tasks

Complete the TODO cells in `assignment.ipynb` (or `assignment.md`). The notebook has three parts:

### Part 1: Data Pipeline

Set up transforms, create datasets, and build DataLoaders.

**Output:** `output/part1_data_summary.json`

### Part 2: Transfer Learning

Load a pretrained model, freeze the backbone, replace the classifier head, and train.

**Outputs:**
- `output/part2_training_history.json`
- `output/part2_training_curves.png`
- `output/part2_model.pt`

### Part 3: Evaluation

Evaluate on the test set and generate metrics.

**Outputs:**
- `output/part3_results.json`
- `output/part3_confusion_matrix.png`

## Running Your Code

```bash
jupytext --to notebook assignment.md
jupyter nbconvert --execute assignment.ipynb
```

Or open `assignment.ipynb` in Jupyter/VS Code and run cells interactively.

## Checking Your Work

```bash
pytest .github/tests/ -v
```

## Output Files Summary

| File | Part | Description |
|:---|:---:|:---|
| `part1_data_summary.json` | 1 | Dataset sizes and class names |
| `part2_training_history.json` | 2 | Loss and accuracy per epoch |
| `part2_training_curves.png` | 2 | Training/validation loss and accuracy plots |
| `part2_model.pt` | 2 | Saved model weights |
| `part3_results.json` | 3 | Test accuracy, precision, recall, F1 |
| `part3_confusion_matrix.png` | 3 | Confusion matrix visualization |

## Hints

- Start with `models.resnet18(weights="DEFAULT")` — it's small and fast
- Use `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` for ImageNet-pretrained models
- Freeze backbone: `for param in model.parameters(): param.requires_grad = False`
- Replace head: `model.fc = nn.Linear(512, num_classes)`
- Train only the head: `optimizer = Adam(model.fc.parameters(), lr=1e-3)`
- 3–5 epochs is enough for a frozen backbone
