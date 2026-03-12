# Part 3: ECG Analysis

## Introduction

In this part, you'll work with the MIT-BIH Arrhythmia Database to build a model for heartbeat classification using a simple neural network architecture. This will help you understand how to apply neural networks to time series data in healthcare.

## Learning Objectives

- Load and preprocess ECG time series data
- Implement a simple neural network for sequence classification
- Train and evaluate the model
- Interpret results in a clinical context

## Setup and Installation

```python
# Install required packages
%pip install -r requirements.txt
%pip install wfdb  # For reading MIT-BIH format

# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wfdb
from scipy import signal
import urllib.request
import zipfile

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configure matplotlib for better visualization
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results/part_3', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

def download_mitbih_dataset():
    """
    Download and extract MIT-BIH Arrhythmia Database.
    
    Returns:
        str: Path to the extracted dataset
    """
    data_dir = 'data/mitdb'
    if os.path.exists(data_dir):
        print("Dataset already downloaded.")
        return data_dir
    
    print("Downloading MIT-BIH Arrhythmia Database...")
    url = "https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
    zip_path = 'data/mitdb.zip'
    
    # Download dataset
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
    
    # Clean up
    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully.")
    return data_dir

# Download dataset
data_dir = download_mitbih_dataset()

# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wfdb
from scipy import signal

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configure matplotlib for better visualization
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results/part_3', exist_ok=True)
os.makedirs('logs', exist_ok=True)
```

## 1. Data Loading and Preprocessing

### Task 1.1: Load ECG Data

```python
def load_ecg_data(record_path):
    """
    Load ECG data from MIT-BIH database.
    
    Args:
        record_path: Path to the record file
    
    Returns:
        signals: ECG signals
        annotations: Beat annotations
    """
    # Read record
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal
    
    # Read annotations
    ann = wfdb.rdann(record_path, 'atr')
    annotations = ann.symbol
    
    return signals, annotations

def verify_data_loading(signals, annotations):
    """
    Verify that the data is loaded correctly.
    
    Args:
        signals: ECG signals
        annotations: Beat annotations
    """
    # Plot sample ECG segment
    plt.figure(figsize=(15, 5))
    plt.plot(signals[:1000, 0])
    plt.title('Sample ECG Segment')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()
    
    # Print data statistics
    print(f"Signal shape: {signals.shape}")
    print(f"Number of annotations: {len(annotations)}")
    print(f"Unique beat types: {np.unique(annotations)}")
```

### Task 1.2: Preprocess the Data
```python
def preprocess_ecg(signals, annotations, window_size=180):
    """
    Preprocess ECG data for model input.
    
    Args:
        signals: ECG signals
        annotations: Beat annotations
        window_size: Size of the window around each beat
    
    Returns:
        X: Preprocessed signals
        y: Labels
    """
    # Normalize signals
    signals = (signals - np.mean(signals)) / np.std(signals)
    
    # Extract beats
    X = []
    y = []
    
    for i, ann in enumerate(annotations):
        if ann in ['N', 'L', 'R', 'A', 'V']:  # Normal and abnormal beats
            # Get window around beat
            start = max(0, i - window_size//2)
            end = min(len(signals), i + window_size//2)
            
            # Pad if necessary
            if start == 0:
                pad_left = window_size//2 - i
                segment = np.pad(signals[start:end], ((pad_left, 0), (0, 0)))
            elif end == len(signals):
                pad_right = window_size//2 - (len(signals) - i)
                segment = np.pad(signals[start:end], ((0, pad_right), (0, 0)))
            else:
                segment = signals[start:end]
            
            X.append(segment)
            
            # Convert annotation to label
            if ann == 'N':
                y.append(0)  # Normal
            else:
                y.append(1)  # Abnormal
    
    return np.array(X), np.array(y)

def verify_preprocessing(X, y):
    """
    Verify that the preprocessing is correct.
    
    Args:
        X: Preprocessed signals
        y: Labels
    """
    print(f"Data shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Plot sample beats
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.plot(X[i, :, 0])
        plt.title(f'Beat Type: {"Normal" if y[i] == 0 else "Abnormal"}')
        plt.axis('off')
    plt.show()
```

## 2. Model Implementation

### Task 2.1: Create Simple Neural Network
```python
def create_simple_nn(input_shape):
    """
    Create a simple neural network for ECG classification.
    This is similar to the architecture from Part 1.
    
    Args:
        input_shape: Shape of input data
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # Flatten input
        tf.keras.layers.Flatten(input_shape=input_shape),
        
        # Dense layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def verify_model_architecture(model):
    """
    Verify that the model architecture meets requirements.
    
    Args:
        model: Keras model
    """
    model.summary()
    
    # Test model with sample input
    sample_input = tf.random.normal((1, 180, 2))  # window_size x channels
    sample_output = model(sample_input)
    print(f"\nSample output shape: {sample_output.shape}")
    
    # Verify architecture requirements
    assert any('dense' in layer.name for layer in model.layers), "Model must include dense layers"
    assert any('dropout' in layer.name for layer in model.layers), "Model must include dropout"
    assert model.loss == 'binary_crossentropy', "Model must use binary crossentropy loss"
    assert any('auc' in metric.name for metric in model.metrics), "Model must include AUC metric"
```

## 3. Training and Evaluation

### Task 3.1: Training Function
```python
def train_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    Train the model and save it.
    
    Args:
        model: Keras model
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        model_name: Name for saving the model
    
    Returns:
        Training history
    """
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'models/{model_name}.keras',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks
    )
    
    return history

def verify_training(history):
    """
    Verify that the training was successful.
    
    Args:
        history: Training history
    """
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

### Task 3.2: Evaluation Function
```python
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model and save metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels
        model_name: Name for saving metrics
    
    Returns:
        Dictionary of metrics
    """
    # Evaluate model
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)
    
    # Get predictions
    predictions = model.predict(X_test)
    predicted_labels = (predictions > 0.5).astype(int)
    
    # Calculate confusion matrix
    cm = tf.math.confusion_matrix(y_test, predicted_labels)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.numpy().ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Save metrics
    metrics = {
        'model': model_name,
        'accuracy': float(test_accuracy),
        'auc': float(test_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.numpy().tolist()
    }
    
    # Save to file
    os.makedirs('results/part_3', exist_ok=True)
    with open(f'results/part_3/{model_name}_metrics.txt', 'w') as f:
        f.write(f"model: {model_name}\n")
        f.write(f"accuracy: {metrics['accuracy']}\n")
        f.write(f"auc: {metrics['auc']}\n")
        f.write(f"precision: {metrics['precision']}\n")
        f.write(f"recall: {metrics['recall']}\n")
        f.write(f"f1_score: {metrics['f1_score']}\n")
        f.write(f"confusion_matrix: {metrics['confusion_matrix']}\n")
        f.write("----\n")
    
    return metrics

def verify_evaluation(metrics):
    """
    Verify that the evaluation is complete.
    
    Args:
        metrics: Dictionary of metrics
    """
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
```

## 4. Main Execution

```python
# Main execution
if __name__ == "__main__":
    # 1. Load and preprocess data
    record_path = 'mitdb/100'  # Example record
    signals, annotations = load_ecg_data(record_path)
    verify_data_loading(signals, annotations)
    
    # Preprocess data
    X, y = preprocess_ecg(signals, annotations)
    verify_preprocessing(X, y)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 2. Create and verify model
    model = create_simple_nn(input_shape=(180, 2))
    verify_model_architecture(model)
    
    # 3. Train and evaluate
    model_name = 'ecg_classifier'
    history = train_model(model, X_train, y_train, X_val, y_val, model_name)
    verify_training(history)
    
    # Save model
    model.save(f'models/{model_name}.keras')
    
    # Evaluate and save metrics
    metrics = evaluate_model(model, X_test, y_test, model_name)
    verify_evaluation(metrics)
```

## Progress Checkpoints

1. **Data Loading**:
   - [ ] Successfully download MIT-BIH dataset
   - [ ] Load and visualize ECG signals
   - [ ] Verify signal shape and annotations

2. **Preprocessing**:
   - [ ] Normalize signals
   - [ ] Extract beat windows
   - [ ] Verify window shapes and labels

3. **Model Implementation**:
   - [ ] Create simple neural network
   - [ ] Verify model architecture
   - [ ] Test model output shape

4. **Training**:
   - [ ] Train model with callbacks
   - [ ] Monitor training progress
   - [ ] Save best model

5. **Evaluation**:
   - [ ] Calculate performance metrics
   - [ ] Save metrics in correct format
   - [ ] Visualize results

## Intended Endpoint

1. **Model Performance**:
   - Accuracy > 75% on test set
   - AUC > 0.80
   - F1-score > 0.70

2. **Required Files**:
   - `models/ecg_classifier.keras`
   - `results/part_3/ecg_classifier_metrics.txt`

3. **Metrics Format**:
   ```
   model: ecg_classifier
   accuracy: float
   auc: float
   precision: float
   recall: float
   f1_score: float
   confusion_matrix: [[TN, FP], [FN, TP]]
   ----
   ```

4. **Model Architecture**:
   - Must use at least 2 dense layers
   - Must include dropout layers
   - Must use binary crossentropy loss
   - Must include AUC metric
