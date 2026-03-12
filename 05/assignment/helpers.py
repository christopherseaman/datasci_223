"""
Helper functions for Classification Assignment.

These are provided for you - you don't need to modify this file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix


def load_fashion_mnist():
    """
    Load Fashion-MNIST dataset and return flattened images with labels.

    Returns
    -------
    X : np.ndarray
        Flattened images, shape (70000, 784), dtype float32
    y : np.ndarray
        Labels, shape (70000,), values 0-9
    """
    from sklearn.datasets import fetch_openml

    # Fetch Fashion-MNIST from OpenML (lighter than TensorFlow)
    mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')

    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    return X, y


def plot_roc_curves(trained_models, X_test, y_test, save_path):
    """
    Generate and save ROC curves for multiple trained models.

    Parameters
    ----------
    trained_models : dict
        Dictionary of {model_name: fitted_model}
    X_test : array-like
        Test features
    y_test : array-like
        Test labels (binary: 0 or 1)
    save_path : str
        Path to save the plot (e.g., 'output/roc_curves.png')
    """
    plt.figure(figsize=(8, 6))

    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Model Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """
    Generate and save a confusion matrix plot.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list
        Display labels for the confusion matrix (e.g., ['Clothing', 'Footwear'])
    save_path : str
        Path to save the plot (e.g., 'output/confusion_matrix.png')
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# Label mappings for reference
FASHION_MNIST_LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
