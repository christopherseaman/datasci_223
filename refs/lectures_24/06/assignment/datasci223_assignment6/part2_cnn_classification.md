# Part 2: CNN Classification

## Introduction

In this part, you'll implement a Convolutional Neural Network (CNN) for EMNIST character recognition. You can choose between TensorFlow/Keras or PyTorch for implementation. This will help you understand CNNs and their advantages for image classification tasks.

## Learning Objectives

- Implement a CNN using either TensorFlow/Keras or PyTorch
- Apply convolutional layers, pooling, and batch normalization
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
os.makedirs('results/part_2', exist_ok=True)
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
    
    # Reshape for CNN input (samples, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
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

### Task 2.1: Create CNN (TensorFlow/Keras)
```python
def create_cnn_keras(input_shape, num_classes):
    """
    Create a CNN using TensorFlow/Keras.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

### Task 2.2: Create CNN (PyTorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Dense layers
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten and dense layers
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def create_cnn_pytorch(num_classes):
    """
    Create a CNN using PyTorch.
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        PyTorch model
    """
    model = CNN(num_classes)
    return model

def verify_model_architecture(model, framework='keras'):
    """
    Verify that the model architecture meets requirements.
    
    Args:
        model: Keras or PyTorch model
        framework: 'keras' or 'pytorch'
    """
    if framework == 'keras':
        # Print model summary
        model.summary()
        
        # Verify architecture requirements
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        pool_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.MaxPooling2D)]
        bn_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
        
        assert len(conv_layers) >= 2, "Model must have at least 2 convolutional layers"
        assert len(pool_layers) >= 1, "Model must have at least 1 pooling layer"
        assert len(bn_layers) >= 1, "Model must have at least 1 batch normalization layer"
        assert model.loss == 'categorical_crossentropy', "Model must use categorical crossentropy loss"
        
        # Test model with sample input
        sample_input = tf.random.normal((1, 28, 28, 1))
        sample_output = model(sample_input)
        print(f"\nSample output shape: {sample_output.shape}")
    
    else:  # PyTorch
        # Print model summary
        print(model)
        
        # Verify architecture requirements
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        pool_layers = [m for m in model.modules() if isinstance(m, nn.MaxPool2d)]
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        
        assert len(conv_layers) >= 2, "Model must have at least 2 convolutional layers"
        assert len(pool_layers) >= 1, "Model must have at least 1 pooling layer"
        assert len(bn_layers) >= 1, "Model must have at least 1 batch normalization layer"
        
        # Test model with sample input
        sample_input = torch.randn(1, 1, 28, 28)
        sample_output = model(sample_input)
        print(f"\nSample output shape: {sample_output.shape}")
```

## 3. Training and Evaluation

### Task 3.1: Training Function (TensorFlow/Keras)
```python
def train_model_keras(model, x_train, y_train, x_val, y_val):
    """
    Train the Keras model and save it.
    
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
            'models/cnn_keras.keras',
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

### Task 3.2: Training Function (PyTorch)
```python
def train_model_pytorch(model, x_train, y_train, x_val, y_val, device='cuda'):
    """
    Train the PyTorch model and save it.
    
    Args:
        model: PyTorch model
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        device: Device to use for training
    
    Returns:
        Training history
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Convert data to PyTorch tensors
    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.LongTensor(np.argmax(y_train, axis=1)).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.LongTensor(np.argmax(y_val, axis=1)).to(device)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(20):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for i in range(0, len(x_train), 32):
            batch_x = x_train[i:i+32]
            batch_y = y_train[i:i+32]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss = train_loss / (len(x_train) / 32)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(0, len(x_val), 32):
                batch_x = x_val[i:i+32]
                batch_y = y_val[i:i+32]
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss = val_loss / (len(x_val) / 32)
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/cnn_pytorch.pt')
            # Save architecture
            with open('models/cnn_pytorch_arch.txt', 'w') as f:
                f.write(str(model))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return history

def verify_training(history, framework='keras'):
    """
    Verify that the training was successful.
    
    Args:
        history: Training history
        framework: 'keras' or 'pytorch'
    """
    if framework == 'keras':
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
    
    else:  # PyTorch
        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history['train_acc'], label='Training')
        ax1.plot(history['val_acc'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history['train_loss'], label='Training')
        ax2.plot(history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

### Task 3.3: Evaluation Function
```python
def evaluate_model(model, x_test, y_test, framework='keras'):
    """
    Evaluate model and save metrics.
    
    Args:
        model: Trained model
        x_test: Test data
        y_test: Test labels
        framework: 'keras' or 'pytorch'
    
    Returns:
        Dictionary of metrics
    """
    if framework == 'keras':
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        
        # Get predictions
        predictions = model.predict(x_test)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)
    
    else:  # PyTorch
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Convert data to PyTorch tensors
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(np.argmax(y_test, axis=1)).to(device)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            outputs = model(x_test)
            test_loss = nn.CrossEntropyLoss()(outputs, y_test).item()
            _, predicted_labels = outputs.max(1)
            test_accuracy = predicted_labels.eq(y_test).sum().item() / len(y_test)
            true_labels = y_test.cpu().numpy()
            predicted_labels = predicted_labels.cpu().numpy()
    
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
        'model': f'cnn_{framework}',
        'accuracy': float(test_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.numpy().tolist()
    }
    
    # Save to file
    os.makedirs('results/part_2', exist_ok=True)
    with open(f'results/part_2/cnn_{framework}_metrics.txt', 'w') as f:
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
    # Choose framework
    framework = 'keras'  # or 'pytorch'
    
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
    if framework == 'keras':
        model = create_cnn_keras(input_shape=(28, 28, 1), num_classes=26)
    else:
        model = create_cnn_pytorch(num_classes=26)
    verify_model_architecture(model, framework)
    
    # 3. Train and evaluate
    if framework == 'keras':
        history = train_model_keras(model, x_train, y_train, x_val, y_val)
    else:
        history = train_model_pytorch(model, x_train, y_train, x_val, y_val)
    verify_training(history, framework)
    
    # Evaluate and save metrics
    metrics = evaluate_model(model, x_test, y_test, framework)
    verify_evaluation(metrics)
```

## Progress Checkpoints

1. **Data Loading**:
   - [ ] Successfully load EMNIST dataset
   - [ ] Verify data shapes and ranges
   - [ ] Visualize sample images

2. **Preprocessing**:
   - [ ] Normalize pixel values
   - [ ] Reshape data for CNN input
   - [ ] Convert labels to one-hot encoding

3. **Model Implementation**:
   - [ ] Choose framework (Keras or PyTorch)
   - [ ] Create model with required layers
   - [ ] Verify architecture requirements

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
   - Accuracy > 85% on test set
   - Must use at least 2 convolutional layers
   - Must include pooling and batch normalization
   - Must use categorical crossentropy loss

2. **Required Files**:
   For Keras:
   - `models/cnn_keras.keras`
   - `results/part_2/cnn_keras_metrics.txt`
   
   For PyTorch:
   - `models/cnn_pytorch.pt`
   - `models/cnn_pytorch_arch.txt`
   - `results/part_2/cnn_pytorch_metrics.txt`

3. **Metrics Format**:
   ```
   model: cnn_{framework}
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
   - Solution: Check internet connection and framework installation

2. **Preprocessing Issues**:
   - Problem: Shape mismatch in CNN layers
   - Solution: Ensure data is properly reshaped to (n_samples, height, width, channels)
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
