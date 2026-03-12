# Demo 3: Imbalanced Classification & Model Interpretation (Disease Prediction)

This demo addresses a common challenge in health data: **imbalanced classes**. Often, the condition we want to predict (e.g., a rare disease) is much less frequent than the negative case. Standard models trained on imbalanced data tend to perform poorly on the minority class because they can achieve high accuracy simply by predicting the majority class most of the time.

We will:
1. Generate synthetic imbalanced data mimicking a rare disease prediction task, including a categorical feature.
2. Handle the categorical feature using `OneHotEncoder`.
3. Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the *training* data.
4. Train a classifier (RandomForest) on the balanced training data.
5. Evaluate the model, focusing on metrics sensitive to imbalance (Recall, F1-score, AUC).
6. Use **eli5** and **SHAP** to interpret the model's feature importances.

## 0. Setup: Install Required Packages 📦

<!---
Make sure all required packages are installed before running the notebook.
The --quiet flag reduces output noise during installation.
--->

```python
%pip install -r requirements.txt --quiet
```

## 1. Setup: Import Libraries

We import necessary libraries:
- Standard data science tools: `numpy`, `pandas`, `matplotlib`, `seaborn`.
- `sklearn.datasets`: To generate synthetic data.
- `sklearn.model_selection`: For splitting data.
- `sklearn.preprocessing`: For `OneHotEncoder`.
- `sklearn.ensemble`: For `RandomForestClassifier`.
- `sklearn.metrics`: For evaluation, including `roc_curve`.
- `imblearn.over_sampling`: For `SMOTE` (install via `pip install imbalanced-learn`).
- `eli5`: For model interpretation (install via `pip install eli5`).
Setting the random seed ensures reproducibility.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve # Added roc_curve
from imblearn.over_sampling import SMOTE # Needs: pip install imbalanced-learn
import eli5 # Needs: pip install eli5

# Configure plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('pastel')
np.random.seed(42) # for reproducibility
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore future warnings from eli5/sklearn
```

## 2. Generate Synthetic Imbalanced Data with Categorical Feature

We use `make_classification` again, but this time create a highly imbalanced dataset (`weights=[0.95, 0.05]`) representing a rare condition (5% positive cases). We also add a synthetic categorical feature ('Region') to demonstrate encoding.

```python
# Generate imbalanced data
X, y = make_classification(
    n_samples=2000,
    n_features=10,       # More features
    n_informative=5,     # Only 5 features are actually useful
    n_redundant=2,
    n_clusters_per_class=2, # Make classes slightly harder to separate
    weights=[0.95, 0.05], # Highly imbalanced: 95% class 0, 5% class 1 (rare disease)
    flip_y=0.01,         # Low noise
    random_state=42
)

# Create DataFrame
feature_names = [f'cont_feat_{i+1}' for i in range(10)]
df = pd.DataFrame(X, columns=feature_names)

# Add a synthetic categorical feature
regions = ['North', 'South', 'East', 'West']
df['Region'] = np.random.choice(regions, size=df.shape[0], p=[0.3, 0.2, 0.25, 0.25])

df['RareDisease'] = y # Target variable

display("Dataset Shape:", df.shape)
display("\nClass distribution (RareDisease):")
target_counts = df['RareDisease'].value_counts()
display(target_counts)
display(target_counts / len(df)) # Print proportions
display("\nCategorical feature distribution (Region):")
display(df['Region'].value_counts())
display("\nFirst 5 rows:")
display(df.head())

# Visualize the imbalance using counts
plt.figure(figsize=(6, 4))
sns.countplot(x='RareDisease', data=df)
plt.title('Initial Class Distribution (0=No Disease, 1=Rare Disease)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['No Disease (0)', 'Rare Disease (1)'])
plt.show()
```

## 3. Split Data into Training and Testing Sets

**Crucially**, we split the data *before* applying SMOTE or encoding. This prevents "data leakage," where information from the test set inadvertently influences the training process (e.g., SMOTE creating synthetic samples based on test data patterns, or the encoder learning categories only present in the test set). The test set must remain completely unseen until final evaluation.

```python
# Separate features (X) and target (y)
X = df.drop('RareDisease', axis=1)
y = df['RareDisease']

## 4. Preprocessing: One-Hot Encode Categorical Feature FIRST

Machine learning models need numerical input. We convert the 'Region' column into numerical format using `OneHotEncoder` BEFORE splitting the data, following the recommended order.

```python
# Identify categorical and numerical columns
categorical_cols = ['Region']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Initialize OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform on the ENTIRE dataset
encoder.fit(X[categorical_cols])
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
X_encoded_array = encoder.transform(X[categorical_cols])
X_encoded = pd.DataFrame(X_encoded_array, columns=encoded_feature_names, index=X.index)

# Combine numerical and encoded categorical features
X_processed = pd.concat([X[numerical_cols], X_encoded], axis=1)

# Store final feature names
final_feature_names = list(X_processed.columns)

display("Processed features shape:", X_processed.shape)
display("\nFirst 5 rows of processed data:")
display(X_processed.head())

# NOW split the processed data
X_train_processed, X_test_processed, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=0.2,
    random_state=42,
    stratify=y # Essential for imbalanced data!
)

display("Processed Training set shape:", X_train_processed.shape)
display("Processed Testing set shape:", X_test_processed.shape)
display("\nTraining set class distribution:")
display(y_train.value_counts(normalize=True))
display("\nTesting set class distribution:")
display(y_test.value_counts(normalize=True))
```

## 5. Handle Imbalance with SMOTE (on Training Data Only!)

Now we address the class imbalance using SMOTE. SMOTE synthetically creates new examples of the minority class (RareDisease=1) by interpolating between existing minority class samples.
**Crucially, SMOTE is applied *only* to the processed training data (`X_train_processed`, `y_train`).** Applying it to the test data would artificially inflate performance metrics because the model would be evaluated on synthetic data it helped create.

First, let's visualize the original training data distribution using two features to see the imbalance.

```python
# Visualize original training data distribution (using first two continuous features)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_processed.iloc[:, 0], y=X_train_processed.iloc[:, 1], hue=y_train, alpha=0.6, palette='pastel')
plt.title(f'Original Training Data Distribution ({X_train_processed.columns[0]} vs {X_train_processed.columns[1]})')
plt.xlabel(X_train_processed.columns[0])
plt.ylabel(X_train_processed.columns[1])
plt.legend(title='RareDisease', labels=['No (0)', 'Yes (1)'])
plt.show()

display("\nOriginal training distribution (counts):")
display(y_train.value_counts())

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE ONLY to the training data
display("\nApplying SMOTE to training data...")
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
display("SMOTE application complete.")

display("\nResampled training distribution (counts):")
display(y_train_resampled.value_counts()) # Should be equal counts
display("\nResampled training features shape:", X_train_resampled.shape)

# Visualize the resampled training data distribution
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_resampled.iloc[:, 0], y=X_train_resampled.iloc[:, 1], hue=y_train_resampled, alpha=0.6, palette='pastel')
plt.title(f'Resampled Training Data Distribution After SMOTE ({X_train_processed.columns[0]} vs {X_train_processed.columns[1]})')
plt.xlabel(X_train_processed.columns[0])
plt.ylabel(X_train_processed.columns[1])
plt.legend(title='RareDisease', labels=['No (0)', 'Yes (1)'])
plt.show()
```

## 6. Train a RandomForest Classifier

We train a RandomForestClassifier on the **resampled (balanced) training data**. This ensures the model learns from both classes adequately, rather than being biased towards the original majority class.

```python
# Initialize RandomForest
rf_smote = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
# Using class_weight='balanced' as an additional measure, though SMOTE is primary

# Train on the RESAMPLED training data
display("Training RandomForest on SMOTE-resampled data...")
rf_smote.fit(X_train_resampled, y_train_resampled)
display("Model training complete.")
```

## 7. Evaluate the Model (on Original Test Set)

We evaluate the model trained on balanced data using the **original, processed test set** (`X_test_processed`, `y_test`). This reflects real-world performance on unseen, imbalanced data.

Pay close attention to:
- **Confusion Matrix:** How many actual rare disease cases (True Positives) did we catch? How many did we miss (False Negatives)?
- **Classification Report:** Specifically, look at the **Recall** for the minority class (RareDisease=1). High recall here means the model is good at identifying the rare cases, which is often the primary goal in imbalanced health problems. Also check the F1-score for a balance between precision and recall.
- **AUC & ROC Curve:** Provide threshold-independent measures of model separability.

```python
# Make predictions on the ORIGINAL (but processed) test set
y_pred_smote = rf_smote.predict(X_test_processed)
y_pred_proba_smote = rf_smote.predict_proba(X_test_processed)[:, 1] # Probabilities for AUC/ROC

# --- Evaluation ---
display("\n--- Evaluation on Original Test Set (Model trained on SMOTE data) ---")

# Accuracy
accuracy_smote = accuracy_score(y_test, y_pred_smote)
display(f"Accuracy: {accuracy_smote:.4f}") # Note: Accuracy might not be the best metric here!

# Confusion Matrix
cm_smote = confusion_matrix(y_test, y_pred_smote)
display("\nConfusion Matrix:")
# display(cm_smote)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Disease (0)', 'Rare Disease (1)'],
            yticklabels=['No Disease (0)', 'Rare Disease (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Model with SMOTE)')
plt.show()


# Classification Report - Focus on Recall for class 1!
report_smote = classification_report(y_test, y_pred_smote, target_names=['No Disease (0)', 'Rare Disease (1)'])
display("\nClassification Report:")
display(report_smote)

# AUC
auc_smote = roc_auc_score(y_test, y_pred_proba_smote)
display(f"\nAUC: {auc_smote:.4f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_smote)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'RandomForest with SMOTE (AUC = {auc_smote:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing') # Diagonal line
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()
```

## 8. Model Interpretation with eli5

eli5 helps explain the model's decisions. `eli5.explain_weights` displays the features ranked by their importance (using permutation importance or impurity-based importance for tree models). This helps identify which factors (original continuous features or encoded regions) the model found most predictive.

We pass the trained model (`rf_smote`) and the final feature names.

```python
display("\n--- eli5 Feature Importances ---")

# Show feature importances calculated by eli5
# For RandomForest, eli5 typically uses feature impurity by default.
explanation = eli5.explain_weights(rf_smote, feature_names=final_feature_names, top=15)

# Format as HTML with custom styling for black text
from eli5.formatters import format_as_html
from IPython.display import HTML

html_explanation = format_as_html(explanation)
styled_html = f'<div style="color: black !important;">{html_explanation}</div>'
display("Displaying top 15 feature importances:")
display(HTML(styled_html))
```

## 9. Interpretation & Conclusion

Let's analyze the results:
- **Impact of SMOTE:** Compare the recall for the 'Rare Disease (1)' class in the classification report to what you might expect without SMOTE (likely much lower). SMOTE significantly improves the model's ability to detect the minority class, even if overall accuracy doesn't change much (or even slightly decreases). The scatter plots visually demonstrate how SMOTE adds synthetic minority samples to balance the feature space representation in the training data.
- **Evaluation Metrics:** For imbalanced problems, focus on Recall (Sensitivity) for the minority class, F1-score, and AUC/ROC curve. High recall means fewer missed cases of the rare disease. The ROC curve visually confirms the model's ability to separate classes better than random guessing.
- **eli5 Interpretation:** The `show_weights` output reveals which features contributed most to the RandomForest's predictions. This could highlight key continuous variables or even indicate if certain regions (from the one-hot encoded features) were associated with the rare disease in our synthetic data. Understanding feature importance is vital for validating if the model is learning meaningful patterns or relying on artifacts.

This demo showed how to handle categorical features and class imbalance—two critical steps in real-world health data modeling. Applying SMOTE correctly (only on training data) and interpreting the model helps build effective and trustworthy classifiers.

**🧠 Comprehension Checkpoint:**

1.  Why is it crucial to apply SMOTE *only* to the training data?
2.  Why might accuracy be a misleading metric for highly imbalanced datasets? Which metrics are often more informative?
3.  What kind of information does `eli5.show_weights` provide about the RandomForest model?
4.  If a feature like `Region_East` appeared high in the eli5 importance list, what might that suggest in the context of this synthetic dataset?

Answers:
1. To prevent data leakage. Applying SMOTE before splitting would mean synthetic samples based on test set information could end up in the training set, artificially inflating performance on the test set. The test set must represent unseen, real-world (imbalanced) data.
2. A model can achieve high accuracy by simply predicting the majority class all the time. Metrics like Recall (for the minority class), F1-score, Precision-Recall curves, and AUC/ROC are more informative as they assess the model's ability to handle the minority class.
3. It ranks the features based on how much they contribute to the model's predictions (e.g., based on how much they reduce impurity across all trees in the forest). It shows which features the model found most useful for classification.
4. It would suggest that belonging to the 'East' region was found by the model to be a relatively strong predictor (either positively or negatively associated) of the 'RareDisease' outcome in this synthetic data, compared to other features.