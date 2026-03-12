# Hints

Use these hints if you get stuck. Try to work through the problem yourself first!

---

## Part 1 Hints

### Filtering data
```python
# Create a boolean mask for labels you want
mask = np.isin(y, [0, 1])  # True where y is 0 or 1
X_filtered = X[mask]
y_filtered = y[mask]
```

### Stratified split
```python
# stratify=y ensures class proportions are preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered,
    test_size=0.2,
    random_state=42,
    stratify=y_filtered
)
```

### Scaling properly
```python
# Fit scaler on training data ONLY, then transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit AND transform
X_test_scaled = scaler.transform(X_test)        # transform only (no fit!)
```

---

## Part 2 Hints

### Encoding labels for XGBoost
```python
# XGBoost requires labels to be 0, 1, 2, ... (not 5, 7, 9)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_filtered)  # [5,7,9] -> [0,1,2]

# Use y_encoded for train_test_split and model training
```

### Cross-validation loop structure
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    for fold, score in enumerate(scores, 1):
        cv_results.append({'model': name, 'fold': fold, 'score': score})

cv_df = pd.DataFrame(cv_results)
```

### Finding the best model
```python
# Group by model and get mean score
mean_scores = cv_df.groupby('model')['score'].mean()

# Get name of model with highest mean score
best_model_name = mean_scores.idxmax()
```

### Using classification_report
```python
# output_dict=True gives you a dictionary instead of a string
report = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame for saving
report_df = pd.DataFrame(report).transpose()
```

---

## Part 3 Hints

### Creating binary labels
```python
# First, filter out bags (label 8)
mask = y != 8
X_filtered = X[mask]
y_original = y[mask]

# Then create binary labels
footwear_labels = {5, 7, 9}
y_binary = np.array([1 if label in footwear_labels else 0 for label in y_original])
```

### Same CV splitter for fair comparison
```python
# Create ONE splitter and use it for ALL models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# This ensures all models are evaluated on the exact same data splits
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
```

### Getting probabilities for ROC curves
```python
# predict_proba returns probabilities for each class
# [:, 1] gets the probability of the positive class (footwear)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Use y_prob (not y_pred) for roc_auc_score
auc = roc_auc_score(y_test, y_prob)
```

### Training all models for ROC plot
```python
# You need ALL models trained (not just the best) to plot ROC curves
trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

# Then pass the dict to the helper function
plot_roc_curves(trained_models, X_test_scaled, y_test, save_path)
```

---

## General Hints

### Saving DataFrames to CSV
```python
# Save without the index column
df.to_csv("output/filename.csv", index=False)

# For classification_report, you might want to keep the index (class names)
report_df.to_csv("output/filename.csv")  # index=True is default
```

### Creating a single-row DataFrame
```python
# For Part 3 test results
test_results = pd.DataFrame([{
    'model': best_model_name,
    'accuracy': test_accuracy,
    'auc': test_auc
}])
```

### Vectorized binary label creation (alternative)
```python
# Instead of a list comprehension, you can use np.isin:
y_binary = np.isin(y_original, [5, 7, 9]).astype(int)
```
