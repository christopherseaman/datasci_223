import os
import pytest


# Helper to parse metrics file
def parse_metrics(filepath):
    metrics = {}
    with open(filepath) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                # Try to parse as float or list
                if v.startswith("["):
                    import ast

                    metrics[k] = ast.literal_eval(v)
                else:
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        metrics[k] = v
    return metrics


def test_part1_files_exist():
    assert os.path.exists("models/emnist_classifier.keras")
    assert os.path.exists("results/part_1/emnist_classifier_metrics.txt")


def test_part1_metrics_format():
    metrics = parse_metrics("results/part_1/emnist_classifier_metrics.txt")
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
    ]:
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_part2_files_exist():
    # Check for either Keras or PyTorch outputs
    keras = os.path.exists("models/cnn_keras.keras")
    pytorch = os.path.exists("models/cnn_pytorch.pt") and os.path.exists(
        "models/cnn_pytorch_arch.txt"
    )
    assert keras or pytorch
    if keras:
        assert os.path.exists("results/part_2/cnn_keras_metrics.txt")
    if pytorch:
        assert os.path.exists("results/part_2/cnn_pytorch_metrics.txt")


def test_part2_metrics_format():
    if os.path.exists("results/part_2/cnn_keras_metrics.txt"):
        metrics = parse_metrics("results/part_2/cnn_keras_metrics.txt")
    else:
        metrics = parse_metrics("results/part_2/cnn_pytorch_metrics.txt")
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
    ]:
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_part3_files_exist():
    # Accept any model type for part 3
    found = False
    for f in os.listdir("models"):
        if f.startswith("ecg_classifier_") and f.endswith(".keras"):
            found = True
    assert found
    found = False
    for f in os.listdir("results/part_3"):
        if f.startswith("ecg_classifier_") and f.endswith("_metrics.txt"):
            found = True
    assert found


def test_part3_metrics_format():
    # Find the metrics file
    for f in os.listdir("results/part_3"):
        if f.endswith("_metrics.txt"):
            metrics = parse_metrics(os.path.join("results/part_3", f))
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "confusion_matrix",
            ]:
                assert key in metrics
            assert 0.0 <= metrics["accuracy"] <= 1.0
            break
    else:
        pytest.skip("No part 3 metrics file found")
