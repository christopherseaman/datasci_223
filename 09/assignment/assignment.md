---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: percent
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# %% [markdown]
# # Assignment 9: Medical Image Classifier
#
# Build a chest X-ray classifier using PyTorch transfer learning.
# We'll use CIFAR-10 with 2 classes for quick iteration — the
# pipeline transfers directly to real medical imaging data.

# %% [markdown]
# ## Setup

# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("output", exist_ok=True)

# %% [markdown]
# ---
# ## Part 1: Data Pipeline
#
# Set up transforms, create datasets, and build DataLoaders.

# %% [markdown]
# ### TODO 1a: Define Transforms
#
# Create two transform pipelines:
# - `train_transform`: resize to 224×224, add augmentation, convert to tensor,
#   normalize with ImageNet stats
# - `eval_transform`: resize to 224×224, convert to tensor, normalize

# %%
# TODO: Define train_transform and eval_transform
train_transform = ...  # Your code here
eval_transform = ...   # Your code here

# %% [markdown]
# ### TODO 1b: Load Dataset and Create Splits
#
# Load CIFAR-10, filter to 2 classes (labels 0 and 1), and split into
# train/val/test sets.

# %%
# Load CIFAR-10
full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
full_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_transform)

# Filter to 2 classes
def filter_two_classes(dataset, classes=(0, 1)):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)

train_data = filter_two_classes(full_train)
test_data = filter_two_classes(full_test)

# TODO: Split train_data into train_set (80%) and val_set (20%)
# Hint: use random_split with a fixed seed for reproducibility
train_set = ...  # Your code here
val_set = ...    # Your code here

# %% [markdown]
# ### TODO 1c: Create DataLoaders

# %%
# TODO: Create DataLoaders with batch_size=32
train_loader = ...  # Your code here
val_loader = ...    # Your code here
test_loader = ...   # Your code here

# %%
# Save data summary
data_summary = {
    "train_size": len(train_set),
    "val_size": len(val_set),
    "test_size": len(test_data),
    "num_classes": 2,
    "class_names": ["airplane", "automobile"],
}
with open("output/part1_data_summary.json", "w") as f:
    json.dump(data_summary, f, indent=2)
print("Saved output/part1_data_summary.json")
print(data_summary)

# %% [markdown]
# ---
# ## Part 2: Transfer Learning
#
# Load a pretrained model, freeze the backbone, replace the head, and train.

# %% [markdown]
# ### TODO 2a: Load and Modify Pretrained Model
#
# Load pretrained ResNet-18, freeze the backbone, and replace `model.fc`
# with a new classification head for 2 classes.

# %%
# TODO: Load pretrained ResNet-18
model = ...  # Your code here

# TODO: Freeze all backbone parameters
# Your code here

# TODO: Replace model.fc with a new head for 2 classes
# Your code here

model = model.to(device)

# Print parameter counts
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# %% [markdown]
# ### TODO 2b: Train the Model
#
# Write a training loop. Train for 5 epochs, tracking loss and accuracy.

# %%
criterion = nn.CrossEntropyLoss()

# TODO: Create an optimizer that trains ONLY the new head parameters
optimizer = ...  # Your code here

num_epochs = 5
history = {"train_loss": [], "val_loss": [], "val_acc": []}

# TODO: Write the training loop
# For each epoch:
#   1. Training phase: model.train(), iterate train_loader, compute loss, backprop
#   2. Validation phase: model.eval(), torch.no_grad(), iterate val_loader
#   3. Record metrics in history dict
for epoch in range(num_epochs):
    # Your code here
    pass

# %%
# Save training history
with open("output/part2_training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print("Saved output/part2_training_history.json")

# %%
# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"], label="Validation")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss Curves")
ax1.legend()

ax2.plot(history["val_acc"], label="Validation", color="green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Validation Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("output/part2_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved output/part2_training_curves.png")

# %%
# Save model
torch.save(model.state_dict(), "output/part2_model.pt")
print("Saved output/part2_model.pt")

# %% [markdown]
# ---
# ## Part 3: Evaluation
#
# Evaluate the trained model on the test set.

# %% [markdown]
# ### TODO 3: Evaluate on Test Set
#
# Run inference on the test set, compute accuracy, and generate a confusion
# matrix.

# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# TODO: Run inference on test_loader, collect predictions and labels
model.eval()
all_preds = []
all_labels = []

# Your code here

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# %%
# Classification report
class_names = ["airplane", "automobile"]
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
print(classification_report(all_labels, all_preds, target_names=class_names))

# Save results
results = {
    "accuracy": float(report["accuracy"]),
    "precision": float(report["weighted avg"]["precision"]),
    "recall": float(report["weighted avg"]["recall"]),
    "f1": float(report["weighted avg"]["f1-score"]),
}
with open("output/part3_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved output/part3_results.json")
print(results)

# %%
# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("output/part3_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved output/part3_confusion_matrix.png")
