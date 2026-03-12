# Assignment 6: Neural Network Showdown — Solution

## Setup

```python
%pip install -q -r requirements.txt

# GPU acceleration (platform-specific)
import platform
if platform.system() == "Darwin" and platform.machine() == "arm64":
    %pip install -q tensorflow-metal

%reset -f
```

```python
import os
import json
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)

# Report available accelerators
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU acceleration: {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"  {gpu.name}")
else:
    print("No GPU detected — using CPU")

from tensorflow import keras
from keras import Sequential
from keras.layers import (
    Dense, Flatten, Dropout, Conv2D, MaxPooling2D, LSTM, Input
)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix

from helpers import (
    load_cifar10, load_ecg5000,
    plot_training_history, plot_confusion_matrix,
    plot_sample_images, plot_ecg_traces, plot_predictions,
    CIFAR10_CLASSES, ECG_CLASSES,
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Setup complete!")
```

---

## Part 1: Dense Baseline on CIFAR-10

```python
print("Part 1: Dense Baseline on CIFAR-10")
print("-" * 40)

X_train, y_train, X_test, y_test = load_cifar10()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

```python
plot_sample_images(X_train, y_train, CIFAR10_CLASSES)
```

```python
model_dense = Sequential([
    Input(shape=(32, 32, 3)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(10, activation="softmax"),
])
```

```python
model_dense.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

```python
early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history_dense = model_dense.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
)
```

```python
test_loss, test_acc = model_dense.evaluate(X_test, y_test, verbose=0)

y_pred = np.argmax(model_dense.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
```

```python
results = {
    "accuracy": float(test_acc),
    "confusion_matrix": cm.tolist(),
}
with open(os.path.join(OUTPUT_DIR, "part1_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"Dense accuracy: {test_acc:.4f}")
print("Saved output/part1_results.json")
```

---

## Part 2: CNN on CIFAR-10

```python
print("\nPart 2: CNN on CIFAR-10")
print("-" * 40)
```

```python
model_cnn = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax"),
])
```

```python
model_cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

```python
callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    ModelCheckpoint(
        "output/best_cnn.keras",
        save_best_only=True,
        monitor="val_accuracy",
    ),
]

history_cnn = model_cnn.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
)
```

```python
plot_training_history(
    history_cnn, os.path.join(OUTPUT_DIR, "part2_training_history.png")
)
```

```python
cnn_loss, cnn_acc = model_cnn.evaluate(X_test, y_test, verbose=0)

y_pred_cnn = np.argmax(model_cnn.predict(X_test, verbose=0), axis=1)
y_true_cnn = np.argmax(y_test, axis=1)
cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
```

```python
results_cnn = {
    "accuracy": float(cnn_acc),
    "confusion_matrix": cm_cnn.tolist(),
}
with open(os.path.join(OUTPUT_DIR, "part2_results.json"), "w") as f:
    json.dump(results_cnn, f, indent=2)

comparison = pd.DataFrame([
    {"model": "Dense", "accuracy": float(test_acc)},
    {"model": "CNN", "accuracy": float(cnn_acc)},
])
comparison.to_csv(os.path.join(OUTPUT_DIR, "part2_comparison.csv"), index=False)

print(f"CNN accuracy:   {cnn_acc:.4f}")
print(f"Dense accuracy: {test_acc:.4f}")
print(f"Improvement:    {cnn_acc - test_acc:+.4f}")
print("Saved output/part2_results.json and output/part2_comparison.csv")
```

---

## Part 3: LSTM on ECG5000

```python
print("\nPart 3: LSTM on ECG5000")
print("-" * 40)

X_train_ecg, y_train_ecg, X_test_ecg, y_test_ecg = load_ecg5000()
print(f"Train: {X_train_ecg.shape}, Test: {X_test_ecg.shape}")
print(f"Classes: {list(ECG_CLASSES.values())}")
```

```python
plot_ecg_traces(X_train_ecg, y_train_ecg, ECG_CLASSES)
```

```python
model_lstm = Sequential([
    Input(shape=(140, 1)),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(5, activation="softmax"),
])
```

```python
model_lstm.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

```python
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history_lstm = model_lstm.fit(
    X_train_ecg, y_train_ecg,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
)
```

```python
plot_training_history(
    history_lstm, os.path.join(OUTPUT_DIR, "part3_training_history.png")
)
```

```python
lstm_loss, lstm_acc = model_lstm.evaluate(X_test_ecg, y_test_ecg, verbose=0)

y_pred_ecg = np.argmax(model_lstm.predict(X_test_ecg, verbose=0), axis=1)
y_true_ecg = np.argmax(y_test_ecg, axis=1)
cm_ecg = confusion_matrix(y_true_ecg, y_pred_ecg)
```

```python
results_ecg = {
    "accuracy": float(lstm_acc),
    "confusion_matrix": cm_ecg.tolist(),
}
with open(os.path.join(OUTPUT_DIR, "part3_results.json"), "w") as f:
    json.dump(results_ecg, f, indent=2)

print(f"LSTM accuracy: {lstm_acc:.4f}")
print("Saved output/part3_results.json")
```

---

## Validation

```python
print("\nAll parts complete!")
print("Run 'pytest .github/tests/ -v' in your terminal to check your work.")
```
