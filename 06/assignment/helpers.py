"""
Helper functions for Neural Network Assignment.
These are provided for you — do not modify this file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


CIFAR10_CLASSES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

ECG_CLASSES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Premature Ventricular",
    3: "Fusion",
    4: "Unknown",
}


def load_cifar10():
    """Load CIFAR-10, normalize pixel values to [0, 1], one-hot encode labels.

    Returns
    -------
    X_train : ndarray, shape (50000, 32, 32, 3), float32 in [0, 1]
    y_train : ndarray, shape (50000, 10), one-hot encoded
    X_test  : ndarray, shape (10000, 32, 32, 3), float32 in [0, 1]
    y_test  : ndarray, shape (10000, 10), one-hot encoded
    """
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, y_train, X_test, y_test


def load_ecg5000(data_dir="data"):
    """Load ECG5000 heartbeat dataset.

    Downloads from data.badmath.org if not cached locally. Combines the
    original train/test splits, reshuffles with a fixed seed, and returns
    an 80/20 split shaped for RNN input.

    Returns
    -------
    X_train : ndarray, shape (~4000, 140, 1), float32
    y_train : ndarray, shape (~4000, 5), one-hot encoded
    X_test  : ndarray, shape (~1000, 140, 1), float32
    y_test  : ndarray, shape (~1000, 5), one-hot encoded
    """
    from tensorflow.keras.utils import to_categorical

    npz_path = os.path.join(data_dir, "ecg5000.npz")
    data = np.load(npz_path)
    train_data = data["train"]
    test_data = data["test"]

    # Combine and reshuffle (original split is 500/4500)
    all_data = np.vstack([train_data, test_data])
    rng = np.random.RandomState(42)
    rng.shuffle(all_data)

    split = int(0.8 * len(all_data))
    X_train = all_data[:split, 1:].astype("float32")
    y_train_raw = (all_data[:split, 0] - 1).astype(int)  # labels 1-5 → 0-4
    X_test = all_data[split:, 1:].astype("float32")
    y_test_raw = (all_data[split:, 0] - 1).astype(int)

    # Reshape for RNN: (samples, timesteps, features)
    X_train = X_train.reshape(-1, 140, 1)
    X_test = X_test.reshape(-1, 140, 1)

    y_train = to_categorical(y_train_raw, 5)
    y_test = to_categorical(y_test_raw, 5)

    return X_train, y_train, X_test, y_test


def plot_training_history(history, save_path):
    """Plot training/validation accuracy and loss curves side by side.

    Parameters
    ----------
    history : keras History object or dict with keys accuracy, val_accuracy,
              loss, val_loss
    save_path : str
    """
    hist = history.history if hasattr(history, "history") else history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(hist["accuracy"], label="Train", linewidth=2)
    ax1.plot(hist["val_accuracy"], label="Validation", linewidth=2)
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(hist["loss"], label="Train", linewidth=2)
    ax2.plot(hist["val_loss"], label="Validation", linewidth=2)
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save a confusion matrix.

    Parameters
    ----------
    y_true : array-like of int, true class indices
    y_pred : array-like of int, predicted class indices
    class_names : list of str
    save_path : str
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sample_images(X, y, class_names=None, n=20, cols=5, save_path=None):
    """Display a grid of sample images with labels.

    Useful for verifying data loaded correctly.

    Parameters
    ----------
    X : ndarray, shape (N, 32, 32, 3), pixel values in [0, 1]
    y : ndarray, shape (N, num_classes), one-hot encoded labels
    class_names : dict mapping int to str (default: CIFAR10_CLASSES)
    n : int, number of images to show
    cols : int, columns in the grid
    save_path : str, optional path to save the figure
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES
    labels = np.argmax(y, axis=1)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(X[i])
        ax.set_title(class_names[labels[i]], fontsize=10)
        ax.axis("off")
    plt.suptitle("Sample Images", fontsize=13)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_ecg_traces(X, y, class_names=None, n_per_class=1, save_path=None):
    """Plot example ECG traces for each heartbeat class.

    Parameters
    ----------
    X : ndarray, shape (N, 140, 1), ECG voltage time series
    y : ndarray, shape (N, 5), one-hot encoded labels
    class_names : dict mapping int to str (default: ECG_CLASSES)
    n_per_class : int, traces to show per class
    save_path : str, optional path to save the figure
    """
    if class_names is None:
        class_names = ECG_CLASSES
    labels = np.argmax(y, axis=1)
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 1, figsize=(12, 2.5 * n_classes), sharex=True)
    for cls in range(n_classes):
        idx = np.where(labels == cls)[0]
        for j in range(min(n_per_class, len(idx))):
            axes[cls].plot(X[idx[j]].flatten(), linewidth=1, alpha=0.8)
        axes[cls].set_title(f"Class {cls}: {class_names[cls]}", fontsize=11)
        axes[cls].set_ylabel("Voltage")
        axes[cls].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time Step")
    plt.suptitle("ECG Heartbeat Types", fontsize=13)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_predictions(X, y_true, y_pred, class_names, n=12, cols=4, save_path=None):
    """Show images with predicted labels, color-coded green (correct) / red (wrong).

    Parameters
    ----------
    X : ndarray, shape (N, 32, 32, 3), images in [0, 1]
    y_true : array-like of int, true class indices
    y_pred : array-like of int, predicted class indices
    class_names : dict mapping int to str
    n : int, number of images to show
    cols : int, columns in the grid
    save_path : str, optional path to save the figure
    """
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(X[i])
        correct = y_pred[i] == y_true[i]
        color = "green" if correct else "red"
        ax.set_title(
            f"{class_names[y_pred[i]]} (true: {class_names[y_true[i]]})",
            color=color, fontsize=9,
        )
        ax.axis("off")
    plt.suptitle("Green = correct, Red = wrong", fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
