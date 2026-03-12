# Part 1: Neural Networks Basics

## Introduction

In this part, you'll implement a simple neural network for EMNIST character recognition. This will help you understand the fundamentals of neural networks, including dense layers, activation functions, and dropout.

## Learning Objectives

- Load and preprocess EMNIST dataset
- Implement a simple neural network with dense layers
- Train and evaluate the model
- Save model and metrics in the correct format

## Setup and Installation

```python
# Install required packages
%pip install -r requirements.txt

# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configure matplotlib for better visualization
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results/part_1', exist_ok=True)
os.makedirs('logs', exist_ok=True)
```

## 1. Data Loading and Preprocessing

### Task 1.1: Load EMNIST Data
```python
def load_emnist_data():
    """
    Load EMNIST dataset for character recognition.
    
    Returns:
        (x_train, y_train), (x_test, y_test): Training and test data
    """
    # Load EMNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.emnist.load_data('letters')
    
    # Print dataset information
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return (x_train, y_train), (x_test, y_test)

def verify_data_loading(x_train, y_train, x_test, y_test):
    """
    Verify that the data is loaded correctly.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
    """
    # Plot sample images
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_train[i].T, cmap='gray')
        plt.title(f'Label: {chr(y_train[i] + 64)}')
        plt.axis('off')
    plt.show()
    
    # Print data statistics
    print(f"Training data range: [{x_train.min()}, {x_train.max()}]")
    print(f"Test data range: [{x_test.min()}, {x_test.max()}]")
    print(f"Label distribution: {np.bincount(y_train)}")
```

### Task 1.2: Preprocess the Data
```python
def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Preprocess EMNIST data for model input.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
    
    Returns:
        Preprocessed data and labels
    """
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for dense layers
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes=26)
    y_test = tf.keras.utils.to_categorical(y_test - 1, num_classes=26)
    
    return (x_train, y_train), (x_test, y_test)

def verify_preprocessing(x_train, y_train, x_test, y_test):
    """
    Verify that the preprocessing is correct.
    
    Args:
        x_train: Preprocessed training images
        y_train: Preprocessed training labels
        x_test: Preprocessed test images
        y_test: Preprocessed test labels
    """
    print(f"Preprocessed training data shape: {x_train.shape}")
    print(f"Preprocessed test data shape: {x_test.shape}")
    print(f"Training data range: [{x_train.min()}, {x_train.max()}]")
    print(f"Test data range: [{x_test.min()}, {x_test.max()}]")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
```

## 2. Model Implementation

### Task 2.1: Create Simple Neural Network
```python
def create_simple_nn(input_shape, num_classes):
    """
    Create a simple neural network for EMNIST classification.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # Dense layers
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def verify_model_architecture(model):
    """
    Verify that the model architecture meets requirements.
    
    Args:
        model: Keras model
    """
    # Print model summary
    model.summary()
    
    # Verify architecture requirements
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
    
    assert len(dense_layers) >= 2, "Model must have at least 2 dense layers"
    assert len(dropout_layers) >= 1, "Model must have at least 1 dropout layer"
    assert model.loss == 'categorical_crossentropy', "Model must use categorical crossentropy loss"
    
    # Test model with sample input
    sample_input = tf.random.normal((1, 784))  # 28x28 = 784
    sample_output = model(sample_input)
    print(f"\nSample output shape: {sample_output.shape}")
```

## 3. Training and Evaluation

### Task 3.1: Training Function
```python
def train_model(model, x_train, y_train, x_val, y_val):
    """
    Train the model and save it.
    
    Args:
        model: Keras model
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
    
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
            'models/emnist_classifier.keras',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
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
def evaluate_model(model, x_test, y_test):
    """
    Evaluate model and save metrics.
    
    Args:
        model: Trained Keras model
        x_test: Test data
        y_test: Test labels
    
    Returns:
        Dictionary of metrics
    """
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    
    # Get predictions
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    
    # Calculate confusion matrix
    cm = tf.math.confusion_matrix(true_labels, predicted_labels)
    
    # Calculate additional metrics
    tn = np.sum(np.diag(cm))
    fp = np.sum(cm) - tn
    fn = np.sum(cm) - tn
    tp = np.sum(cm) - tn - fp - fn
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Save metrics
    metrics = {
        'model': 'emnist_classifier',
        'accuracy': float(test_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.numpy().tolist()
    }
    
    # Save to file
    os.makedirs('results/part_1', exist_ok=True)
    with open('results/part_1/emnist_classifier_metrics.txt', 'w') as f:
        f.write(f"model: {metrics['model']}\n")
        f.write(f"accuracy: {metrics['accuracy']}\n")
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
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
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
    (x_train, y_train), (x_test, y_test) = load_emnist_data()
    verify_data_loading(x_train, y_train, x_test, y_test)
    
    # Preprocess data
    (x_train, y_train), (x_test, y_test) = preprocess_data(x_train, y_train, x_test, y_test)
    verify_preprocessing(x_train, y_train, x_test, y_test)
    
    # Split training data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    # 2. Create and verify model
    model = create_simple_nn(input_shape=(784,), num_classes=26)
    verify_model_architecture(model)
    
    # 3. Train and evaluate
    history = train_model(model, x_train, y_train, x_val, y_val)
    verify_training(history)
    
    # Save model
    model.save('models/emnist_classifier.keras')
    
    # Evaluate and save metrics
    metrics = evaluate_model(model, x_test, y_test)
    verify_evaluation(metrics)
```

## Progress Checkpoints

1. **Data Loading**:
   - [ ] Successfully load EMNIST dataset
   - [ ] Verify data shapes and ranges
   - [ ] Visualize sample images

2. **Preprocessing**:
   - [ ] Normalize pixel values
   - [ ] Reshape data for dense layers
   - [ ] Convert labels to one-hot encoding

3. **Model Implementation**:
   - [ ] Create model with required layers
   - [ ] Verify architecture requirements
   - [ ] Test model with sample input

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
   - Accuracy > 80% on test set
   - Must use at least 2 dense layers
   - Must include dropout layers
   - Must use categorical crossentropy loss

2. **Required Files**:
   - `models/emnist_classifier.keras`
   - `results/part_1/emnist_classifier_metrics.txt`

3. **Metrics Format**:
   ```
   model: emnist_classifier
   accuracy: float
   precision: float
   recall: float
   f1_score: float
   confusion_matrix: [[TN, FP], [FN, TP]]
   ----
   ```

## Common Issues and Solutions

1. **Data Loading Issues**:
   - Problem: EMNIST dataset not found
   - Solution: Check internet connection and TensorFlow installation

2. **Preprocessing Issues**:
   - Problem: Shape mismatch in dense layers
   - Solution: Ensure data is properly reshaped to (n_samples, 784)
   - Problem: Label encoding errors
   - Solution: Verify label range and one-hot encoding

3. **Model Issues**:
   - Problem: Training instability
   - Solution: Add batch normalization, reduce learning rate
   - Problem: Overfitting
   - Solution: Increase dropout, use data augmentation

4. **Evaluation Issues**:
   - Problem: Metrics format incorrect
   - Solution: Follow the exact format specified
   - Problem: Performance below threshold
   - Solution: Adjust architecture, hyperparameters
