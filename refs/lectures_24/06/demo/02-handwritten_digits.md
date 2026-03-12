---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Classification of Handwritten Digits

This demo shows how to build and compare multiple machine learning models on the same dataset. We're using handwritten digits (0 and 1) as a simple example, but the same approach can be applied to health data classification problems like disease diagnosis or risk stratification.

## Setup

First, we need to install and import the necessary libraries. For machine learning projects, we typically need:
- Data manipulation: numpy, pandas
- Visualization: matplotlib, seaborn
- Machine learning: scikit-learn, xgboost
- Deep learning: tensorflow

```python
# Install required packages (once per virtual environment)
%pip install -q numpy pandas matplotlib seaborn scikit-learn tensorflow tensorflow-datasets xgboost

# If apple silicon install tensorflow-metal
import os

if os.uname().machine == "arm64":
    %pip install -q tensorflow-macos tensorflow-metal
    pass

%reset -f
```

```python
# Import packages
import os
import string
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from IPython.display import display, Markdown

# ML packages
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# XGBoost
from xgboost import XGBClassifier
# Deep Learning
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Constants
IMAGE_SIZE = 28  # Size of each image (28x28 pixels)
REBUILD = True   # Whether to rebuild models or use cached results
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
```

## Helper Functions

We'll define several helper functions to make our code more organized and reusable:

```python
# Define helper functions
def int_to_char(label):
    """
    Convert an integer label to the corresponding character.
    
    Args:
        label: Integer label from the EMNIST dataset
        
    Returns:
        String representation of the label (digit or letter)
    """
    if label < 10:
        return str(label)  # Digits 0-9
    elif label < 36:
        return chr(label - 10 + ord('A'))  # Uppercase letters A-Z
    else:
        return chr(label - 36 + ord('a'))  # Lowercase letters a-z

def show_image(row):
    """
    Display a single image and its corresponding label.
    
    Args:
        row: DataFrame row containing 'image' and 'label' fields
    """
    image = row['image']
    label = row['label']
    plt.imshow(image, cmap='gray')
    plt.title('Label: ' + int_to_char(label))
    plt.axis('off')
    plt.show()

def show_grid(data, title=None, num_cols=5, figsize=(20, 10)):
    """
    Display a grid of images from the dataset.
    
    Args:
        data: DataFrame containing 'image' and 'label' columns
        title: Optional title for the plot
        num_cols: Number of columns in the grid
        figsize: Size of the figure
    """
    num_images = len(data)
    num_rows = (num_images - 1) // num_cols + 1
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if title is not None:
        fig.suptitle(title, fontsize=16)
        
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_images:
                axes[i, j].imshow(data.iloc[index]['image'], cmap='gray')
                axes[i, j].axis('off')
                label = int_to_char(data.iloc[index]['label'])
                axes[i, j].set_title(label)
    plt.show()

def plot_training_history(history, metric='accuracy'):
    """
    Plot the training and validation metrics during model training.
    
    Args:
        history: History object returned by model.fit()
        metric: Metric to plot ('accuracy' or 'loss')
    """
    if metric == 'accuracy':
        train_metric = history.history['accuracy']
        val_metric = history.history['val_accuracy']
        y_label = 'Accuracy'
        title = 'Training and Validation Accuracy'
    else:  # 'loss'
        train_metric = history.history['loss']
        val_metric = history.history['val_loss']
        y_label = 'Loss'
        title = 'Training and Validation Loss'
    
    epochs = range(1, len(train_metric) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metric, 'bo-', label=f'Training {metric}')
    plt.plot(epochs, val_metric, 'r-', label=f'Validation {metric}')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def display_model_performance(task, model_name, metrics_dict):
    """
    Display performance metrics and confusion matrix for a model.
    
    Args:
        task: Task name (e.g., '0_vs_1')
        model_name: Name of the model
        metrics_dict: Dictionary containing metrics for all models
    """
    # Create DataFrames to display metrics and confusion matrix
    metrics_df = pd.DataFrame()
    cm_df = pd.DataFrame()
    
    # Extract metrics for the specified model
    for key, value in metrics_dict[task][model_name].items():
        # Only process the confusion matrix as a DataFrame
        if key == 'confusion_matrix' and type(value) == np.ndarray:
            cm_df = pd.DataFrame(value,
                                index=['actual 0', 'actual 1'],
                                columns=['predicted 0', 'predicted 1'])
        # Skip other array data (fpr, tpr) from display
        elif key not in ['fpr', 'tpr'] and not isinstance(value, np.ndarray):
            metrics_df[key] = [value]
    
    # Display results
    display(Markdown(f'# Performance Metrics: {model_name}'))
    display(metrics_df)
    display(Markdown(f'# Confusion Matrix: {model_name}'))
    display(cm_df)
```

## Loading and Preparing Data

The EMNIST (Extended MNIST) dataset contains handwritten characters including digits, uppercase letters, and lowercase letters. It's a great dataset for practicing image classification.

In health data science, image classification is used in many applications:
- Classifying medical images (X-rays, MRIs, CT scans)
- Identifying cell types in microscopy images
- Detecting abnormalities in pathology slides

```python
# Load EMNIST dataset using TensorFlow Datasets
print("Loading EMNIST dataset...")
# Load the 'byclass' split which includes digits and letters
emnist_dataset, emnist_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True,  # Returns tuple (img, label) instead of dict
    with_info=True,      # Includes dataset metadata
)

# Get the training and test datasets
train_ds, test_ds = emnist_dataset
```

## Preprocessing the Images

Next, we need to preprocess the images to make them suitable for machine learning models. This includes normalizing pixel values to the range [0, 1].

```python
# Function to preprocess images
def preprocess_images(image, label):
    # Convert to float and normalize to [0, 1]
    # This is important because neural networks work better with normalized data
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing to both training and test datasets
train_ds = train_ds.map(preprocess_images)
test_ds = test_ds.map(preprocess_images)
```

## Converting to Numpy Arrays

TensorFlow datasets are efficient but for this demo, we'll convert to numpy arrays for easier manipulation.

```python
# Convert to numpy arrays for easier manipulation
train_images = []
train_labels = []
for image, label in tfds.as_numpy(train_ds.take(10000)):  # Limit to 10000 samples for speed
    train_images.append(image)
    train_labels.append(label)

test_images = []
test_labels = []
for image, label in tfds.as_numpy(test_ds.take(2000)):  # Limit to 2000 samples for speed
    test_images.append(image)
    test_labels.append(label)
```

## Creating DataFrames for Analysis

We'll organize our data in pandas DataFrames, which makes it easier to manipulate and visualize.

```python
# Create DataFrame to store training data
train_data = pd.DataFrame()
train_data['image'] = train_images  # Original 28x28 images
train_data['image_flat'] = [img.reshape(-1) for img in train_images]  # Flattened to 784 pixels
train_data['label'] = train_labels  # Numeric labels
train_data['class'] = [int_to_char(label) for label in train_labels]  # Character labels

# Create DataFrame to store validation/test data
valid_data = pd.DataFrame()
valid_data['image'] = test_images  # Original 28x28 images
valid_data['image_flat'] = [img.reshape(-1) for img in test_images]  # Flattened to 784 pixels
valid_data['label'] = test_labels  # Numeric labels
valid_data['class'] = [int_to_char(label) for label in test_labels]  # Character labels
```

## Visualize Some Examples

Visualization is a crucial step in any data science project. It helps us understand the data and identify potential issues. For image data, we want to see what the images actually look like.

```python
# Visualize some examples from the dataset

# Display a random image from the training set
random_index = np.random.randint(0, len(train_data))
show_image(train_data.iloc[random_index])

# Show a random set of 25 images in a 5x5 grid
show_grid(train_data.sample(25), title='Random Sample of 25 Images')
```

## Create Binary Classification Dataset (0 vs 1)

While the EMNIST dataset contains many character classes, we'll focus on a simple binary classification problem: distinguishing between digits 0 and 1. This is a good starting point because:

1. It's one of the simplest classification tasks (the shapes are very different)
2. It allows us to use binary classification metrics
3. We can compare different model types on the same straightforward task

Binary classification is common in healthcare applications, such as:
- Disease present/absent in diagnostic tests
- High-risk/low-risk patient stratification
- Treatment effective/ineffective evaluations

```python
# Create a subset with only digits 0 and 1 for binary classification

# Define which symbols to include
digits_to_classify = ['0', '1']

# Filter training data to only include selected digits
train_mask = train_data['class'].apply(lambda x: x in digits_to_classify)
train_binary = train_data[train_mask].reset_index(drop=True)

# Filter validation data to only include selected digits
valid_mask = valid_data['class'].apply(lambda x: x in digits_to_classify)
valid_binary = valid_data[valid_mask].reset_index(drop=True)

# Show some examples of the binary classification dataset
show_grid(train_binary.sample(10), title="Sample of 0's and 1's", num_cols=5)
```

```python
# Initialize metrics dictionary to track model performance
metrics_dict = {
    '0_vs_1': {  # Task name (0 vs 1 classifier)
        'random_forest': {},
        'logistic_regression': {},
        'xgboost': {},
        'neural_network': {}
    }
}
```

## Model 1: Random Forest Classifier

Random Forests are ensemble models that combine multiple decision trees to make predictions. They work by:
1. Creating many decision trees, each trained on a random subset of the data
2. Having each tree "vote" on the classification
3. Taking the majority vote as the final prediction

Key advantages:
- Resistant to overfitting compared to single decision trees
- Can handle high-dimensional data well
- Provides feature importance measures
- Works well "out of the box" with minimal tuning

In healthcare, Random Forests are used for:
- Predicting patient outcomes
- Identifying risk factors for diseases
- Classifying medical images

```python
# Model 1: Random Forest Classifier
task = '0_vs_1'
model_name = 'random_forest'

# Initialize Random Forest classifier with 100 trees
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    random_state=42    # Seed for reproducibility
)

# Train the model
print("Training Random Forest model...")
rf_model.fit(
    train_binary['image_flat'].tolist(),  # Flattened images (features)
    train_binary['label']                 # Labels (0 or 1)
)

# Make predictions on validation data
rf_predictions = rf_model.predict(valid_binary['image_flat'].tolist())

# Calculate performance metrics
accuracy = accuracy_score(valid_binary['label'], rf_predictions)
precision = precision_score(valid_binary['label'], rf_predictions)
recall = recall_score(valid_binary['label'], rf_predictions)
f1 = f1_score(valid_binary['label'], rf_predictions)
conf_matrix = confusion_matrix(valid_binary['label'], rf_predictions)

# Calculate ROC curve and AUC
# For ROC curve, we need probability estimates, not just class predictions
rf_proba = rf_model.predict_proba(valid_binary['image_flat'].tolist())[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(valid_binary['label'], rf_proba)
auc_rf = auc(fpr_rf, tpr_rf)

# Store metrics in dictionary
metrics_dict[task][model_name] = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': conf_matrix,
    'fpr': fpr_rf,
    'tpr': tpr_rf,
    'auc': auc_rf
}

# Display performance metrics
display_model_performance(task, model_name, metrics_dict)
```

## Model 2: Logistic Regression

Logistic Regression is one of the simplest and most interpretable classification algorithms. Despite its name, it's used for classification, not regression.

Key concepts:
- Uses a logistic function to model the probability of a binary outcome
- Creates a linear decision boundary between classes
- Outputs probabilities between 0 and 1
- Simple and computationally efficient

In healthcare, Logistic Regression is used for:
- Predicting disease risk based on patient characteristics
- Analyzing factors that contribute to treatment success
- Identifying biomarkers associated with clinical outcomes

**Important preprocessing step**: Data scaling is critical for logistic regression. Without scaling, features with larger values can dominate the model and prevent convergence.

```python
# Model 2: Logistic Regression
task = '0_vs_1'
model_name = 'logistic_regression'

# Scale the data (important for logistic regression)
# Without scaling, the model might not converge properly
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_binary['image_flat'].tolist())
valid_scaled = scaler.transform(valid_binary['image_flat'].tolist())

# Initialize logistic regression model
lr_model = LogisticRegression(
    max_iter=1000,    # Maximum number of iterations
    random_state=42   # Seed for reproducibility
)

# Train the model
print("Training Logistic Regression model...")
lr_model.fit(
    train_scaled,           # Scaled training data
    train_binary['label']   # Labels (0 or 1)
)

# Make predictions on validation data
lr_predictions = lr_model.predict(valid_scaled)

# Calculate performance metrics
accuracy = accuracy_score(valid_binary['label'], lr_predictions)
precision = precision_score(valid_binary['label'], lr_predictions)
recall = recall_score(valid_binary['label'], lr_predictions)
f1 = f1_score(valid_binary['label'], lr_predictions)
conf_matrix = confusion_matrix(valid_binary['label'], lr_predictions)

# Calculate ROC curve and AUC
lr_proba = lr_model.predict_proba(valid_scaled)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(valid_binary['label'], lr_proba)
auc_lr = auc(fpr_lr, tpr_lr)

# Store metrics in dictionary
metrics_dict[task][model_name] = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': conf_matrix,
    'fpr': fpr_lr,
    'tpr': tpr_lr,
    'auc': auc_lr
}

# Display performance metrics
display_model_performance(task, model_name, metrics_dict)
```

## Model 3: XGBoost Classifier

XGBoost (eXtreme Gradient Boosting) is an advanced implementation of gradient boosting algorithms, known for winning many machine learning competitions.

Key concepts:
- Uses boosting, where each new model corrects errors made by previous models
- Builds trees sequentially, with each new tree focusing on the mistakes of the ensemble so far
- Highly optimized for performance and speed
- Often achieves state-of-the-art results on tabular data

In healthcare applications, XGBoost is frequently used for risk prediction models due to its high accuracy and ability to handle complex relationships in data.

```python
# Model 3: XGBoost Classifier
task = '0_vs_1'
model_name = 'xgboost'

# Initialize XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=100,  # Number of boosting rounds
    random_state=42    # Seed for reproducibility
)

# Train the model
print("Training XGBoost model...")
xgb_model.fit(
    train_binary['image_flat'].tolist(),  # Flattened images (features)
    train_binary['label']                 # Labels (0 or 1)
)

# Make predictions on validation data
xgb_predictions = xgb_model.predict(valid_binary['image_flat'].tolist())

# Calculate performance metrics
accuracy = accuracy_score(valid_binary['label'], xgb_predictions)
precision = precision_score(valid_binary['label'], xgb_predictions)
recall = recall_score(valid_binary['label'], xgb_predictions)
f1 = f1_score(valid_binary['label'], xgb_predictions)
conf_matrix = confusion_matrix(valid_binary['label'], xgb_predictions)

# Calculate ROC curve and AUC
xgb_proba = xgb_model.predict_proba(valid_binary['image_flat'].tolist())[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(valid_binary['label'], xgb_proba)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# Store metrics in dictionary
metrics_dict[task][model_name] = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': conf_matrix,
    'fpr': fpr_xgb,
    'tpr': tpr_xgb,
    'auc': auc_xgb
}

# Display performance metrics
display_model_performance(task, model_name, metrics_dict)
```

## Model 4: Neural Network

Neural Networks are inspired by the structure of the human brain and are particularly well-suited for image classification tasks. They consist of layers of interconnected "neurons" that learn to recognize patterns in data.

Key concepts:
- Input layer receives the raw data (pixel values)
- Hidden layers learn increasingly complex features
- Output layer produces the final classification
- Neurons use activation functions to introduce non-linearity
- Training involves adjusting weights through backpropagation

For image data, neural networks have some specific requirements:
1. Images need to maintain their 2D structure (or be reshaped to do so)
2. Pixel values should be normalized (typically to [0,1] range)

```python
# Model 4: Neural Network
task = '0_vs_1'
model_name = 'neural_network'

# Set random seed for reproducibility
tf.random.set_seed(42)
```

## Preparing Data for the Neural Network

Neural networks typically work with data in a specific format:
1. Reshape images to 28x28x1 (height, width, channels)
2. Normalize pixel values to [0,1] range

```python
# Prepare training data
# Use np.stack to properly combine the arrays into a single multi-dimensional array
train_images_nn = np.stack(train_binary['image'].tolist())
train_labels_nn = np.array(train_binary['label'])

# Prepare validation data
valid_images_nn = np.stack(valid_binary['image'].tolist())
valid_labels_nn = np.array(valid_binary['label'])

# Create a simple neural network model
nn_model = Sequential([
    # Input layer - specify the shape of our images
    # EMNIST images from tfds are in the format (28, 28, 1)
    keras.layers.InputLayer(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),  # Using 'shape' instead of deprecated 'input_shape'
    
    # Flatten the 2D image to a 1D array
    keras.layers.Flatten(),
    
    # Hidden layer with 128 neurons and ReLU activation
    keras.layers.Dense(128, activation='relu'),
    
    # Output layer with sigmoid activation (for binary classification)
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
nn_model.compile(
    optimizer='adam',               # Adam optimization algorithm
    loss='binary_crossentropy',     # Loss function for binary classification
    metrics=['accuracy']            # Track accuracy during training
)

# Display model summary
nn_model.summary()

# Train the model
print("Training Neural Network model...")
history = nn_model.fit(
    train_images_nn,                           # Training images
    train_labels_nn,                           # Training labels
    epochs=10,                                 # Number of training epochs
    validation_data=(valid_images_nn, valid_labels_nn)  # Validation data
)

# Evaluate the model
loss, accuracy = nn_model.evaluate(valid_images_nn, valid_labels_nn)

# Get raw predictions (probabilities)
nn_proba = nn_model.predict(valid_images_nn).flatten()

# Convert to binary predictions
nn_predictions = (nn_proba > 0.5).astype(int)

# Calculate performance metrics
precision = precision_score(valid_labels_nn, nn_predictions)
recall = recall_score(valid_labels_nn, nn_predictions)
f1 = f1_score(valid_labels_nn, nn_predictions)
conf_matrix = confusion_matrix(valid_labels_nn, nn_predictions)

# Calculate ROC curve and AUC
fpr_nn, tpr_nn, _ = roc_curve(valid_labels_nn, nn_proba)
auc_nn = auc(fpr_nn, tpr_nn)

# Store metrics in dictionary
metrics_dict[task][model_name] = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': conf_matrix,
    'fpr': fpr_nn,
    'tpr': tpr_nn,
    'auc': auc_nn
}

# Display performance metrics
display_model_performance(task, model_name, metrics_dict)

# Plot training history
plot_training_history(history, 'accuracy')
plot_training_history(history, 'loss')
```

```python
# Use cached results if not rebuilding models
if not REBUILD:
    # These are pre-computed results to avoid running the models again
    metrics_dict['0_vs_1']['xgboost'] = {
        'accuracy': 0.9994218698381235,
        'precision': 0.999210484762356,
        'recall': 0.9996840442338073,
        'f1': 0.9994472084024323,
        'confusion_matrix': np.array([[5773, 5], [2, 6328]]),
        'fpr': np.array([0.0, 0.0008658009, 1.0]),
        'tpr': np.array([0.0, 0.9996840442, 1.0]),
        'auc': 0.9994091217
    }
    metrics_dict['0_vs_1']['random_forest'] = {
        'accuracy': 0.9992566897918731,
        'precision': 0.9987375729840618,
        'recall': 0.9998420221169037,
        'f1': 0.9992894923817794,
        'confusion_matrix': np.array([[5770, 8], [1, 6329]]),
        'fpr': np.array([0.0, 0.0013852814, 1.0]),
        'tpr': np.array([0.0, 0.9998420221, 1.0]),
        'auc': 0.9992283704
    }
    metrics_dict['0_vs_1']['logistic_regression'] = {
        'accuracy': 0.9972745292368682,
        'precision': 0.9965305156915313,
        'recall': 0.9982622432859399,
        'f1': 0.9973956278115382,
        'confusion_matrix': np.array([[5756, 22], [11, 6319]]),
        'fpr': np.array([0.0, 0.0038116035, 1.0]),
        'tpr': np.array([0.0, 0.9982622433, 1.0]),
        'auc': 0.9972253199
    }
    metrics_dict['0_vs_1']['neural_network'] = {
        'accuracy': 0.9995870590209961,
        'precision': 0.9992107340173638,
        'recall': 1.0,
        'f1': 0.9996052112120015,
        'confusion_matrix': np.array([[5773, 5], [0, 6330]]),
        'fpr': np.array([0.0, 0.0008658009, 1.0]),
        'tpr': np.array([0.0, 1.0, 1.0]),
        'auc': 0.9995670996
    }

# Display all metrics as a raw dictionary
display(metrics_dict)
```

## Compare All Models

When comparing different machine learning models, it's important to look at multiple metrics rather than just accuracy. This is especially true for imbalanced datasets where a model could achieve high accuracy by simply predicting the majority class.

The key metrics we're comparing:
- **Accuracy**: Overall correctness (correct predictions / total predictions)
- **Precision**: When the model predicts "1", how often is it correct? (true positives / predicted positives)
- **Recall**: How many actual "1"s did the model find? (true positives / actual positives)
- **F1 Score**: Harmonic mean of precision and recall, balancing both concerns
- **AUC (Area Under the ROC Curve)**: Measures the model's ability to distinguish between classes across all possible thresholds

ROC curves are particularly valuable in healthcare applications because they:
1. Show the tradeoff between sensitivity and specificity
2. Help determine optimal threshold for clinical decision-making
3. Allow comparison of different diagnostic tests regardless of the threshold chosen

F1 scores are also important in healthcare when you need to balance:
- False positives (which might lead to unnecessary treatment)
- False negatives (which might miss critical diagnoses)

```python
# Create a DataFrame to compare all models
results_df = pd.DataFrame()

# Convert metrics dictionary to DataFrame
for task in metrics_dict:
    # Create a DataFrame from the metrics for this task
    task_df = pd.DataFrame.from_dict(metrics_dict[task], orient='index')
    
    # Add columns for task and model name
    task_df = task_df.assign(task=task, model=task_df.index)
    
    # Set index to task and model
    task_df = task_df.set_index(['task', 'model'])
    
    # Append to main results DataFrame
    results_df = pd.concat([results_df, task_df])

# Display the comparison table
display(results_df)
```

## Model Performance Comparison (Zoomed View)

Let's visualize the performance of each model across different metrics. First, we'll look at a zoomed-in view (0.95-1.0) to see the small differences between models:

```python
# Visualize model performance comparison (zoomed view)

# Define colors for each model
model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create grouped bar chart comparing metrics across models
plt.figure(figsize=(14, 8))

# Define metrics to plot (including F1)
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
models = list(metrics_dict['0_vs_1'].keys())
model_names = [model.replace('_', ' ').title() for model in models]

# Set up the bar positions
x = np.arange(len(metrics_to_plot))
width = 0.2  # Width of each bar
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]  # Offsets for each model's bars

# Plot each model's performance as grouped bars
for i, model in enumerate(models):
    # Extract metric values for this model
    metric_values = [metrics_dict['0_vs_1'][model][metric] for metric in metrics_to_plot]
    
    # Plot as bars
    plt.bar(x + offsets[i], metric_values, width,
            label=model_names[i],
            color=model_colors[i])

# Add value labels on top of each bar
for i, model in enumerate(models):
    metric_values = [metrics_dict['0_vs_1'][model][metric] for metric in metrics_to_plot]
    for j, value in enumerate(metric_values):
        plt.text(x[j] + offsets[i], value + 0.001,
                 f'{value:.3f}',
                 ha='center', va='bottom',
                 fontsize=9, rotation=90)

# Customize the plot
plt.title('Performance Metrics Comparison Across Models (Zoomed View)', fontsize=16)
plt.xticks(x, metrics_to_plot)
plt.ylabel('Score', fontsize=14)
plt.ylim(0.95, 1.0)  # Zoomed view to show differences
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
```

## ROC Curve Comparison

The Receiver Operating Characteristic (ROC) curve is a powerful tool for evaluating and comparing classification models. It plots the True Positive Rate (sensitivity) against the False Positive Rate (1-specificity) at various threshold settings.

Key concepts:
- **Area Under the Curve (AUC)**: A single number summarizing the ROC curve performance (higher is better)
- **Perfect classifier**: AUC = 1.0 (upper left corner)
- **Random classifier**: AUC = 0.5 (diagonal line)

ROC curves are particularly valuable in healthcare applications:
- Diagnostic test evaluation: Finding the optimal threshold for a test
- Comparing different screening methods
- Balancing sensitivity and specificity based on clinical needs

```python
# Create ROC curve comparison plot
plt.figure(figsize=(10, 8))

# Plot ROC curve for each model
for i, model in enumerate(metrics_dict['0_vs_1'].keys()):
    # Get ROC curve data
    fpr = metrics_dict['0_vs_1'][model]['fpr']
    tpr = metrics_dict['0_vs_1'][model]['tpr']
    auc_score = metrics_dict['0_vs_1'][model]['auc']
    
    # Plot ROC curve
    plt.plot(fpr, tpr,
             label=f"{model.replace('_', ' ').title()} (AUC = {auc_score:.4f})",
             color=model_colors[i],
             linewidth=2)

# Plot random classifier line
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')

# Customize the plot
plt.title('ROC Curve Comparison', fontsize=16)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
```

## Model Performance Comparison (Full Range)

After seeing the ROC curves, let's look at the performance metrics on a full 0-1 scale. This view helps us understand that despite the small differences we saw in the zoomed view, all models are performing extremely well on this task:

```python
# Visualize model performance comparison (full range view)

# Define colors for each model
model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create grouped bar chart comparing metrics across models
plt.figure(figsize=(14, 8))

# Define metrics to plot (including F1)
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
models = list(metrics_dict['0_vs_1'].keys())
model_names = [model.replace('_', ' ').title() for model in models]

# Set up the bar positions
x = np.arange(len(metrics_to_plot))
width = 0.2  # Width of each bar
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]  # Offsets for each model's bars

# Plot each model's performance as grouped bars
for i, model in enumerate(models):
    # Extract metric values for this model
    metric_values = [metrics_dict['0_vs_1'][model][metric] for metric in metrics_to_plot]
    
    # Plot as bars
    plt.bar(x + offsets[i], metric_values, width,
            label=model_names[i],
            color=model_colors[i])

# Customize the plot
plt.title('Performance Metrics Comparison Across Models (Full Range)', fontsize=16)
plt.xticks(x, metrics_to_plot)
plt.ylabel('Score', fontsize=14)
plt.ylim(0, 1.0)  # Full range view
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Add a text annotation explaining the high performance
plt.figtext(0.5, 0.01,
            "Note: All models achieve >95% performance on all metrics for this simple task.\n"
            "In real healthcare applications, differences are often more pronounced.",
            ha='center', fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

# Show the plot
plt.tight_layout(rect=[0, 0.08, 1, 1])  # Make room for the annotation
plt.show()
```

## Key Takeaways

1. **Multiple Approaches**: We've seen how different algorithms can solve the same classification problem.

2. **Performance Metrics**: While accuracy is important, both F1 scores and ROC curves provide valuable insights:
   - F1 scores balance precision and recall in a single metric
   - ROC curves and AUC provide a comprehensive evaluation across different threshold settings

3. **Model Complexity**:
   - Random Forest and XGBoost are powerful "out-of-the-box" models
   - Logistic Regression is simpler but requires proper data scaling
   - Neural Networks can be very powerful but require more configuration

4. **Healthcare Applications**: Binary classification is common in healthcare:
   - Disease present/absent in diagnostic tests
   - High-risk/low-risk patient stratification
   - Treatment effective/ineffective evaluations

5. **Choosing the Right Metric for Healthcare**:
   - F1 score: When balancing false positives and false negatives is critical
   - ROC/AUC: When comparing diagnostic tests or finding optimal thresholds
   - Precision: When false positives are costly (e.g., unnecessary treatments)
   - Recall: When false negatives are dangerous (e.g., missing critical diagnoses)

6. **Tips**:
   - Always split your data into training and test sets
   - Scale your data for linear models like Logistic Regression
   - Set random_state for reproducibility
   - Look at multiple metrics, not just accuracy
   - Visualize your data and results
