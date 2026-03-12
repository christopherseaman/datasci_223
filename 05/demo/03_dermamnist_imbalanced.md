# Demo 3: Imbalanced Classification with DermaMNIST

This demo addresses a common challenge in medical data: **class imbalance**. We'll use **DermaMNIST**, a dataset of skin lesion images with 7 classes that are naturally imbalanced - some lesion types are much rarer than others.

## Learning Objectives

By the end of this demo, you will be able to:

1. Identify and visualize class imbalance in real data
2. Apply SMOTE to balance training data
3. Evaluate with metrics appropriate for imbalanced data
4. Interpret models with eli5

## 0. Setup

```python
%pip install -q medmnist scikit-learn imbalanced-learn eli5 matplotlib seaborn pandas numpy
```

## 1. Load DermaMNIST

DermaMNIST contains 10,015 dermatoscopic images of pigmented skin lesions, classified into 7 disease categories.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from medmnist import DermaMNIST
import warnings
warnings.filterwarnings('ignore')

# Download and load the dataset
train_dataset = DermaMNIST(split='train', download=True)
val_dataset = DermaMNIST(split='val', download=True)
test_dataset = DermaMNIST(split='test', download=True)

# Class names
class_names = [
    'Actinic keratoses',      # 0
    'Basal cell carcinoma',   # 1
    'Benign keratosis',       # 2
    'Dermatofibroma',         # 3
    'Melanoma',               # 4
    'Melanocytic nevi',       # 5
    'Vascular lesions'        # 6
]

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"\nImage shape: {train_dataset.imgs[0].shape}")
print(f"\nClasses: {class_names}")
```

## 2. Visualize Class Imbalance

This is a key step - understanding the imbalance before trying to fix it.

```python
# Get labels
y_train = train_dataset.labels.squeeze()
y_val = val_dataset.labels.squeeze()
y_test = test_dataset.labels.squeeze()

# Count samples per class
train_counts = pd.Series(y_train).value_counts().sort_index()

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = plt.cm.Set3(np.linspace(0, 1, 7))
bars = axes[0].bar(range(7), train_counts.values, color=colors)
axes[0].set_xticks(range(7))
axes[0].set_xticklabels([f'{i}\n{name[:15]}...' if len(name) > 15 else f'{i}\n{name}'
                         for i, name in enumerate(class_names)], rotation=0, fontsize=8)
axes[0].set_ylabel('Number of Samples')
axes[0].set_title('Training Set Class Distribution')

# Add count labels on bars
for bar, count in zip(bars, train_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 str(count), ha='center', va='bottom', fontsize=9)

# Pie chart showing imbalance
axes[1].pie(train_counts.values, labels=[f'{i}: {name[:12]}...' if len(name) > 12 else f'{i}: {name}'
                                          for i, name in enumerate(class_names)],
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Class Proportions')

plt.tight_layout()
plt.show()

# Print imbalance ratio
print(f"\nImbalance ratio (max/min): {train_counts.max() / train_counts.min():.1f}x")
print(f"Majority class: {class_names[train_counts.idxmax()]} ({train_counts.max()} samples)")
print(f"Minority class: {class_names[train_counts.idxmin()]} ({train_counts.min()} samples)")
```

## 3. View Sample Images

```python
# Show sample images from each class
fig, axes = plt.subplots(2, 7, figsize=(16, 5))

for class_idx in range(7):
    # Get indices for this class
    indices = np.where(y_train == class_idx)[0]

    # Show 2 samples per class
    for row in range(2):
        if row < len(indices):
            img = train_dataset.imgs[indices[row]]
            axes[row, class_idx].imshow(img)
            if row == 0:
                axes[row, class_idx].set_title(f'{class_idx}: {class_names[class_idx][:12]}',
                                                fontsize=8)
        axes[row, class_idx].axis('off')

plt.suptitle('Sample Images from Each Class', fontsize=12)
plt.tight_layout()
plt.show()
```

## 4. Prepare Data

Flatten images and combine train+val for cross-validation.

```python
from sklearn.preprocessing import StandardScaler

# Flatten images: (N, 28, 28, 3) -> (N, 2352) for RGB
X_train = train_dataset.imgs.reshape(len(train_dataset), -1).astype(np.float32)
X_val = val_dataset.imgs.reshape(len(val_dataset), -1).astype(np.float32)
X_test = test_dataset.imgs.reshape(len(test_dataset), -1).astype(np.float32)

# Combine train + val
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.hstack([y_train, y_val])

print(f"Training features shape: {X_train_full.shape}")
print(f"Test features shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)
```

## 5. Train Without Balancing (Baseline)

First, train a model without addressing imbalance to see the problem.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train baseline model
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline_model.fit(X_train_scaled, y_train_full)

# Predict
y_pred_baseline = baseline_model.predict(X_test_scaled)

# Evaluate
print("BASELINE (No Balancing):")
print("=" * 50)
print(classification_report(y_test, y_pred_baseline, target_names=class_names, zero_division=0))
```

## 6. Apply SMOTE to Balance Classes

SMOTE creates synthetic samples of minority classes by interpolating between existing samples.

```python
from imblearn.over_sampling import SMOTE

# Check original distribution
print("Original training distribution:")
original_counts = pd.Series(y_train_full).value_counts().sort_index()
for i, count in enumerate(original_counts):
    print(f"  {class_names[i]}: {count}")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_full)

# Check new distribution
print(f"\nAfter SMOTE:")
resampled_counts = pd.Series(y_train_resampled).value_counts().sort_index()
for i, count in enumerate(resampled_counts):
    print(f"  {class_names[i]}: {count}")

print(f"\nSamples: {len(y_train_full)} -> {len(y_train_resampled)}")
```

## 7. Train with Balanced Data

```python
# Train on balanced data
balanced_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
balanced_model.fit(X_train_resampled, y_train_resampled)

# Predict
y_pred_balanced = balanced_model.predict(X_test_scaled)

# Evaluate
print("WITH SMOTE BALANCING:")
print("=" * 50)
print(classification_report(y_test, y_pred_balanced, target_names=class_names, zero_division=0))
```

## 8. Compare Results

```python
from sklearn.metrics import balanced_accuracy_score, f1_score

# Calculate metrics
metrics = {
    'Model': ['Baseline', 'SMOTE'],
    'Accuracy': [
        (y_test == y_pred_baseline).mean(),
        (y_test == y_pred_balanced).mean()
    ],
    'Balanced Accuracy': [
        balanced_accuracy_score(y_test, y_pred_baseline),
        balanced_accuracy_score(y_test, y_pred_balanced)
    ],
    'Macro F1': [
        f1_score(y_test, y_pred_baseline, average='macro'),
        f1_score(y_test, y_pred_balanced, average='macro')
    ]
}

comparison_df = pd.DataFrame(metrics)
print("Comparison:")
print(comparison_df.to_string(index=False))

# Visual comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrices
for ax, (title, y_pred) in zip(axes, [('Baseline', y_pred_baseline), ('SMOTE', y_pred_balanced)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=range(7), yticklabels=range(7))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{title} Confusion Matrix')

plt.tight_layout()
plt.show()
```

## 9. Model Interpretation with eli5

Use eli5 to understand which features (pixels) the model uses.

```python
import eli5
from eli5.sklearn import PermutationImportance

# Calculate permutation importance on a sample
sample_size = 500
sample_indices = np.random.choice(len(X_test_scaled), sample_size, replace=False)
X_sample = X_test_scaled[sample_indices]
y_sample = y_test[sample_indices]

# Feature names (pixel positions)
feature_names = [f'pixel_{i}' for i in range(X_train_scaled.shape[1])]

# Show feature importances from eli5
print("Top Feature Importances (Random Forest):")
eli5.show_weights(balanced_model, feature_names=feature_names, top=20)
```

## 10. Visualize Important Regions

```python
# Get feature importances
importances = balanced_model.feature_importances_

# Reshape to image format (28x28x3 RGB)
importance_image = importances.reshape(28, 28, 3)

# Average across color channels for visualization
importance_gray = importance_image.mean(axis=2)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sample lesion image
sample_idx = 0
axes[0].imshow(test_dataset.imgs[sample_idx])
axes[0].set_title(f'Sample Lesion ({class_names[y_test[sample_idx]]})')
axes[0].axis('off')

# Importance heatmap
im = axes[1].imshow(importance_gray, cmap='hot')
axes[1].set_title('Feature Importance Heatmap')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046)

# Overlay
axes[2].imshow(test_dataset.imgs[sample_idx])
axes[2].imshow(importance_gray, cmap='hot', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## Takeaways

- **Accuracy is misleading** for imbalanced data - use balanced accuracy, macro F1, or per-class recall
- **SMOTE only on training data** - never on test data (data leakage)
- **Trade-offs exist** - balancing may slightly reduce overall accuracy but improves minority class detection
- In medical settings, missing a rare but serious condition (like melanoma) can be worse than a false positive
