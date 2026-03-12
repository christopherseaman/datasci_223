# Demo 3: Building Neural Networks from Scratch 🏗️

## Goal
Build and train a CNN from scratch using both PyTorch and Keras, and implement an RNN for sequence data.

## Introduction to Neural Networks for Health Data

In this demo, we'll explore three key types of neural networks commonly used in health data science:

1. **Convolutional Neural Networks (CNNs)** - Perfect for medical imaging data
2. **Recurrent Neural Networks (RNNs)** - Ideal for time series data like ECG signals
3. **Hybrid Architectures** - Combining different network types for complex health data

We'll start with image classification using the EMNIST dataset, then move to sequence data with ECG signals. This progression helps understand how different architectures are suited for different types of health data.

## Setup
```python
# Install required packages
%pip install -q tensorflow torch numpy matplotlib tensorflow-datasets torchvision

# If apple silicon install tensorflow-metal
import os

if os.uname().machine == "arm64":
    %pip install -q tensorflow-macos tensorflow-metal
    pass

%reset -f

# Now import all required packages
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Create directories for saving models and checkpoints
%mkdir -p models
%mkdir -p checkpoints

# Set to True to rebuild the models from scratch
REBUILD = False
```

## Part 1: CNN in PyTorch

### Understanding CNNs for Medical Imaging

Convolutional Neural Networks (CNNs) are particularly powerful for medical imaging because they can:
- Detect patterns at different scales (from small features to large structures)
- Learn hierarchical representations of images
- Handle variations in image orientation and size
- Process multiple channels (e.g., RGB, grayscale, multi-modal medical images)

### Data Loading and Preprocessing
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using EMNIST statistics
])

# Load EMNIST byclass dataset (contains all digits and letters)
train_dataset = torchvision.datasets.EMNIST(
    root='./data', 
    split='byclass',  # Use byclass to get all characters
    train=True,
    download=True, 
    transform=transform
)

# Filter to keep only digits (classes 0-9)
digit_indices = [i for i, (_, label) in enumerate(train_dataset) if label < 10]
train_dataset = torch.utils.data.Subset(train_dataset, digit_indices)

# Same for test set
test_dataset = torchvision.datasets.EMNIST(
    root='./data', 
    split='byclass',
    train=False,
    download=True, 
    transform=transform
)
test_digit_indices = [i for i, (_, label) in enumerate(test_dataset) if label < 10]
test_dataset = torch.utils.data.Subset(test_dataset, test_digit_indices)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=64,  # Process 64 images at a time
    shuffle=True    # Randomize order for better training
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1000,  # Larger batch size for testing
    shuffle=False     # No need to shuffle test data
)

# Visualize some examples
def plot_examples(loader, num_examples=5):
    """Visualize sample images from the dataset."""
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    fig = plt.figure(figsize=(12, 3))
    for i in range(num_examples):
        ax = fig.add_subplot(1, num_examples, i + 1)
        ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_examples(train_loader)
```

### CNN Architecture for Medical Imaging

The CNN architecture we'll build is inspired by medical imaging applications. It includes:
- Convolutional layers to detect features
- Max pooling to reduce spatial dimensions
- Dropout to prevent overfitting
- Fully connected layers for classification

This architecture is similar to those used in medical image analysis tasks like:
- Tumor detection in MRI scans
- Cell classification in microscopy images
- Organ segmentation in CT scans

```python
class CNN(nn.Module):
    """A CNN architecture suitable for medical image classification."""
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # Second convolutional layer: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Dropout layer to prevent overfitting
        self.conv2_drop = nn.Dropout2d()
        # Fully connected layers
        self.fc1 = nn.Linear(1600, 100)  # 1600 = 64 * 5 * 5 (after two max pools)
        self.fc2 = nn.Linear(100, 10)    # 10 output classes (digits 0-9)

    def forward(self, x):
        # First conv block
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        
        # Second conv block
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.conv2_drop(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 1600)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Initialize model, loss function, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters())  # Adam optimizer for training
```

### Training and Evaluation in PyTorch

The training process includes:
- Forward pass: Compute predictions
- Backward pass: Calculate gradients
- Optimization: Update model weights
- Evaluation: Assess model performance

This is similar to how medical imaging models are trained, with careful attention to:
- Validation metrics
- Model checkpointing
- Training history tracking

```python
import pickle
import time

# Define paths for saving model and history
PYTORCH_MODEL_PATH = 'models/mnist_cnn_pytorch.pth'
PYTORCH_HISTORY_PATH = 'models/mnist_cnn_pytorch_history.pkl'

def train(epoch):
    """Train the model for one epoch."""
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear previous gradients
        output = model(data)   # Forward pass
        loss = criterion(output, target)  # Calculate loss
        loss.backward()        # Backward pass
        optimizer.step()       # Update weights
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return train_loss / len(train_loader)

def test():
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')
    return test_loss, accuracy

# Train and evaluate
if REBUILD or not os.path.exists(PYTORCH_MODEL_PATH):
    print("Training new model...")
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    start_time = time.time()
    
    for epoch in range(1, 3):
        train_loss = train(epoch)
        test_loss, test_acc = test()
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_acc)
    
    # Save model and history
    torch.save(model.state_dict(), PYTORCH_MODEL_PATH)
    with open(PYTORCH_HISTORY_PATH, 'wb') as f:
        pickle.dump(history, f)
    
    end_time = time.time()
    print(f"\nTraining time: {end_time - start_time:.2f} seconds")
else:
    print("Loading saved model...")
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
    with open(PYTORCH_HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['test_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['test_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()
```

## Part 2: CNN in Keras

### Keras for Medical Imaging

Keras provides a high-level API that makes it easier to build and train models. This is particularly useful in healthcare applications where:
- Rapid prototyping is important
- Code readability is crucial
- Integration with existing systems is needed

### Data Loading and Preprocessing
```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Load EMNIST byclass dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Filter function to keep only digits (classes 0-9)
def filter_digits(image, label):
    return tf.less(label, 10)

# Preprocess function
def preprocess(image, label):
    # Convert to float and normalize
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128  # Increased batch size for better GPU utilization

# Prepare training dataset with optimized pipeline
ds_train = (ds_train
    .filter(filter_digits)
    .cache()  # Cache the filtered dataset
    .shuffle(10000)  # Shuffle before batching
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

# Prepare test dataset
ds_test = (ds_test
    .filter(filter_digits)
    .cache()
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

# Create a copy of the dataset for visualization
ds_train_viz = ds_train
ds_test_viz = ds_test

# Visualize some examples
def plot_tfds_examples(dataset, num_examples=5):
    """Visualize sample images from the TensorFlow dataset."""
    batch = next(iter(dataset))
    images, labels = batch
    
    plt.figure(figsize=(12, 3))
    for i in range(min(num_examples, len(images))):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(images[i].numpy(), cmap='gray')
        plt.title(f'Label: {labels[i].numpy()}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_tfds_examples(ds_train_viz)
```

### CNN Architecture in Keras

The Keras implementation provides the same functionality as PyTorch but with a more streamlined API. This makes it easier to:
- Experiment with different architectures
- Add new layers
- Modify the training process

```python
# Define paths for saving model and history
KERAS_MODEL_PATH = 'models/mnist_cnn_keras.keras'
KERAS_HISTORY_PATH = 'models/mnist_cnn_keras_history.pkl'

# Define CNN model
model = models.Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for digits
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Train or load model
if REBUILD or not os.path.exists(KERAS_MODEL_PATH):
    print("Training new model...")
    start_time = time.time()
    
    # Train model with callbacks
    # Callbacks are functions that are executed at specific points during training
    # They're particularly useful in healthcare ML for monitoring training, preventing overfitting,
    # and ensuring we save the best model for clinical use
    callbacks = [
        # ModelCheckpoint: Saves the model during training
        # Parameters explained:
        # - KERAS_MODEL_PATH: Where to save the model
        # - save_best_only=True: Only save when the model improves
        # - monitor='val_accuracy': Use validation accuracy to determine "best"
        #
        # In healthcare applications, this ensures we keep the most accurate model
        # for patient data, which is critical for clinical decision support
        tf.keras.callbacks.ModelCheckpoint(
            KERAS_MODEL_PATH,
            save_best_only=True,
            monitor='val_accuracy'
        ),
        
        # EarlyStopping: Stops training when improvement stops
        # Parameters explained:
        # - monitor='val_accuracy': Watch validation accuracy
        # - patience=3: Wait 3 epochs with no improvement before stopping
        # - restore_best_weights=True: Revert to the best weights when stopped
        #
        # This prevents overfitting, which is crucial in healthcare where
        # models must generalize well to new patient data
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        ds_train,
        epochs=5,
        validation_data=ds_test,
        callbacks=callbacks
    )
    
    # Save history
    with open(KERAS_HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    
    end_time = time.time()
    print(f"\nTraining time: {end_time - start_time:.2f} seconds")
else:
    print("Loading saved model...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    with open(KERAS_HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
```

## Part 3: RNN for ECG Signal Classification

### Understanding Time Series Data in Healthcare

Time series data is common in healthcare, including:
- ECG signals
- Vital signs monitoring
- Sleep studies
- Continuous glucose monitoring

RNNs are particularly well-suited for these applications because they can:
- Capture temporal dependencies
- Handle variable-length sequences
- Learn patterns over time

### Data Loading and Preprocessing
```python
import numpy as np
import urllib.request
import zipfile
import os

# Download and extract ECG5000 dataset
if not os.path.exists('ECG5000'):
    print("Downloading ECG5000 dataset...")
    urllib.request.urlretrieve(
        'https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip',
        'ECG5000.zip'
    )
    with zipfile.ZipFile('ECG5000.zip', 'r') as zip_ref:
        zip_ref.extractall('ECG5000')

# Load the data
def load_ecg_data():
    """Load the ECG5000 dataset."""
    # Load training data
    train_data = np.loadtxt('ECG5000/ECG5000_TRAIN.txt')
    X_train = train_data[:, 1:]  # All columns except first
    y_train = train_data[:, 0]   # First column is label
    
    # Load test data
    test_data = np.loadtxt('ECG5000/ECG5000_TEST.txt')
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    
    return X_train, y_train, X_test, y_test

# Load and normalize data
X_train, y_train, X_test, y_test = load_ecg_data()
```


### Data Preprocessing for Time Series

Time series data requires special preprocessing:
- Normalization to handle different scales
- Reshaping for RNN input
- Handling missing values
- Dealing with variable lengths

```python
# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for RNN input (samples, time steps, features)
X_train = X_train.reshape(-1, 140, 1)
X_test = X_test.reshape(-1, 140, 1)
```

### Understanding Model Performance in Healthcare Context

The model's performance metrics have specific implications for healthcare:

1. **Accuracy by Class**
   - Normal beats should have high accuracy
   - Critical conditions (like premature ventricular contractions) need careful monitoring
   - Unknown beats may require human review

2. **Confidence Levels**
   - High confidence in normal beats is expected
   - Lower confidence in rare conditions is acceptable
   - Very low confidence should trigger human review

3. **Clinical Impact**
   - False negatives in critical conditions are more serious
   - False positives may lead to unnecessary interventions
   - Unknown classifications should be flagged for review

This evaluation helps us understand how the model might perform in a real clinical setting and what additional safeguards might be needed.

## Understanding the ECG5000 Dataset

The ECG5000 dataset contains 5 classes of ECG signals:
1. Normal beat
2. Supraventricular premature beat
3. Premature ventricular contraction
4. Fusion of ventricular and normal beat
5. Unknown beat

Each signal is 140 time steps long, representing a single heartbeat. This dataset is particularly valuable for:
- Learning to classify different types of heartbeats
- Understanding temporal patterns in medical signals
- Developing models for real-time monitoring
  
### What Does a Real ECG Look Like?

Before we look at individual heartbeats, let's see what a real, multi-beat ECG signal looks like. In clinical practice, ECGs are recorded as continuous signals over time, showing many heartbeats in sequence. Our dataset breaks these up into single-beat segments for classification, but it's important to see the bigger picture first!

```python
# Plot a multi-beat ECG by concatenating several single beats
def plot_multibeat_ecg(X, y, num_beats=10):
    """
    Plot a continuous ECG signal by concatenating several single-beat segments.
    """
    multi_beat = np.concatenate(X[:num_beats]).ravel()
    plt.figure(figsize=(15, 3))
    plt.plot(multi_beat)
    plt.title(f"Multi-beat ECG (showing {num_beats} consecutive beats)")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.show()

plot_multibeat_ecg(X_train, y_train, num_beats=10)
```

### Visualizing Example Beats from Each Class

Let's look at one example from each class in the dataset. This helps us see the diversity of heartbeat shapes and why classification is challenging.

```python
# Plot one example from each class
unique_classes = np.unique(y_train)
def plot_examples_by_class(X, y):
    plt.figure(figsize=(15, 2 * len(unique_classes)))
    for idx, cls in enumerate(unique_classes):
        i = np.where(y == cls)[0][0]  # First occurrence of this class
        plt.subplot(len(unique_classes), 1, idx + 1)
        plt.plot(X[i])
        plt.title(f"Class {int(cls)}")
        plt.xlabel("Time Step")
        plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

plot_examples_by_class(X_train, y_train)
```

### Model Definition and Training
```python
# Define paths for saving model and history
RNN_MODEL_PATH = 'models/ecg_rnn.keras'
RNN_HISTORY_PATH = 'models/ecg_rnn_history.pkl'

# Define RNN model
model = models.Sequential([
    layers.SimpleRNN(32, input_shape=(140, 1)),
    layers.Dense(5, activation='softmax')  # 5 classes in ECG5000
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train or load RNN model
if REBUILD or not os.path.exists(RNN_MODEL_PATH):
    print("Training new RNN model...")
    start_time = time.time()
    # Callbacks for the RNN model - similar to those used in the CNN model above
    callbacks = [
        # ModelCheckpoint: Saves the best model during training
        # For time series healthcare data like ECG signals, having the most accurate
        # model is essential as misclassifications could lead to missed diagnoses
        tf.keras.callbacks.ModelCheckpoint(
            RNN_MODEL_PATH,
            save_best_only=True,  # Only save when the model improves
            monitor='val_accuracy'  # Use validation accuracy as the metric
        ),
        
        # EarlyStopping: Prevents overfitting by stopping training when no improvement
        # This is especially important for ECG data where the model needs to
        # generalize across different patients with varying heart patterns
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Watch validation accuracy
            patience=3,  # Allow 3 epochs without improvement before stopping
            restore_best_weights=True  # Use the weights from the best epoch
        )
        
        # Note: For clinical applications, we might also consider:
        # - Adding a callback to log predictions for regulatory review
        # - Using class weights to handle imbalanced heart conditions
        # - Implementing custom metrics for specific clinical thresholds
    ]
    history = model.fit(
        X_train, y_train,
        epochs=5,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    # Always save only the dict
    with open(RNN_HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    hist_dict = history.history
    end_time = time.time()
    print(f"\nTraining time: {end_time - start_time:.2f} seconds")
else:
    print("Loading saved RNN model...")
    model = tf.keras.models.load_model(RNN_MODEL_PATH)
    with open(RNN_HISTORY_PATH, 'rb') as f:
        hist_dict = pickle.load(f)

# Create confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for ECG Classification')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# Print classification report (suppress undefined metric warning for clarity)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, zero_division=0))
```

### Note: Model Evaluation in Health Data Science

> If you see a warning like "precision is ill-defined and being set to 0.0 in labels with no predicted samples," it means your model never predicted some classes. This is a common issue in real-world health data:
>
> - **Why does this happen?**
>   - The model may be underfitting (not learning enough).
>   - Some classes are much rarer than others (class imbalance).
>   - The model architecture or training setup may not be suitable.
>
> - **Why does it matter?**
>   - In healthcare, missing rare but important classes (like a dangerous arrhythmia) can have serious consequences.
>   - A model that only predicts the most common class is not clinically useful.
>
> - **What can you do?**
>   - Check the class distribution in your data.
>   - Try more training, a more complex model, or class weighting.
>   - Use evaluation metrics (like confusion matrices) to spot these issues early.
>
> **Bottom line:**  
> Always look beyond overall accuracy. In health data science, it's critical to ensure your model works for all classes, especially the rare and important ones!

### Visualizing Model Predictions: Actual vs. Predicted

The plot below shows a few ECG signals from the test set. Each plot displays:
- The **true class** and the **predicted class**.
- The **confidence** (softmax probability) for the predicted class.
- **Green** lines indicate correct predictions, **red** lines indicate incorrect predictions.

> Note: Confidence is the model's estimated probability for its prediction. High confidence does not always mean the prediction is correct, especially if the model is not well-trained or the data is ambiguous.

```python
def plot_ecg_predictions(X, y, model, num_examples=5):
    """
    Plot ECG signals with their true and predicted labels.
    - Green plot: correct prediction
    - Red plot: incorrect prediction
    - Title shows true label, predicted label, and confidence
    """
    predictions = model.predict(X)
    plt.figure(figsize=(15, 2 * num_examples))
    for i in range(num_examples):
        plt.subplot(num_examples, 1, i+1)
        true_label = int(y[i])
        pred_label = np.argmax(predictions[i])
        confidence = predictions[i][pred_label]
        color = 'green' if true_label == pred_label else 'red'
        plt.plot(X[i], color=color)
        plt.title(
            f"True: {true_label}, Pred: {pred_label} "
            f"(Confidence: {confidence:.2%})",
            color=color
        )
        plt.xlabel("Time Step")
        plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

plot_ecg_predictions(X_test, y_test, model)
```
