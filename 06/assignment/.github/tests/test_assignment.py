"""
Tests for Neural Network Assignment

Tests verify:
1. Output files exist with correct structure
2. Models achieve reasonable performance thresholds
3. CNN outperforms Dense baseline
4. Training history plots were generated
"""

import pytest
import json
import os
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")


class TestPart1:
    """Test Part 1: Dense Baseline on CIFAR-10"""

    def test_part1_output_exists(self):
        """part1_results.json was created."""
        path = os.path.join(OUTPUT_DIR, "part1_results.json")
        assert os.path.exists(path), (
            "part1_results.json should exist. Run the assignment notebook first."
        )

    def test_part1_has_required_keys(self):
        """part1_results.json has accuracy and confusion_matrix."""
        path = os.path.join(OUTPUT_DIR, "part1_results.json")
        if not os.path.exists(path):
            pytest.skip("part1_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert "accuracy" in results, "Results should contain 'accuracy'"
        assert "confusion_matrix" in results, "Results should contain 'confusion_matrix'"

    def test_part1_accuracy_threshold(self):
        """Dense model achieves >20% accuracy on CIFAR-10."""
        path = os.path.join(OUTPUT_DIR, "part1_results.json")
        if not os.path.exists(path):
            pytest.skip("part1_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert results["accuracy"] > 0.20, (
            f"Dense model should achieve >20% accuracy on CIFAR-10 (random is 10%), got {results['accuracy']:.2%}"
        )

    def test_part1_confusion_matrix_shape(self):
        """Confusion matrix is 10x10 for 10-class classification."""
        path = os.path.join(OUTPUT_DIR, "part1_results.json")
        if not os.path.exists(path):
            pytest.skip("part1_results.json not found")

        with open(path) as f:
            results = json.load(f)

        cm = np.array(results["confusion_matrix"])
        assert cm.shape == (10, 10), f"Confusion matrix should be 10x10, got {cm.shape}"


class TestPart2:
    """Test Part 2: CNN on CIFAR-10"""

    def test_part2_output_exists(self):
        """part2_results.json was created."""
        path = os.path.join(OUTPUT_DIR, "part2_results.json")
        assert os.path.exists(path), (
            "part2_results.json should exist. Run the assignment notebook first."
        )

    def test_part2_has_required_keys(self):
        """part2_results.json has accuracy and confusion_matrix."""
        path = os.path.join(OUTPUT_DIR, "part2_results.json")
        if not os.path.exists(path):
            pytest.skip("part2_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert "accuracy" in results, "Results should contain 'accuracy'"
        assert "confusion_matrix" in results, "Results should contain 'confusion_matrix'"

    def test_part2_accuracy_threshold(self):
        """CNN achieves >55% accuracy on CIFAR-10."""
        path = os.path.join(OUTPUT_DIR, "part2_results.json")
        if not os.path.exists(path):
            pytest.skip("part2_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert results["accuracy"] > 0.55, (
            f"CNN should achieve >55% accuracy on CIFAR-10, got {results['accuracy']:.2%}"
        )

    def test_part2_confusion_matrix_shape(self):
        """Confusion matrix is 10x10 for 10-class classification."""
        path = os.path.join(OUTPUT_DIR, "part2_results.json")
        if not os.path.exists(path):
            pytest.skip("part2_results.json not found")

        with open(path) as f:
            results = json.load(f)

        cm = np.array(results["confusion_matrix"])
        assert cm.shape == (10, 10), f"Confusion matrix should be 10x10, got {cm.shape}"

    def test_part2_comparison_exists(self):
        """part2_comparison.csv was created."""
        path = os.path.join(OUTPUT_DIR, "part2_comparison.csv")
        assert os.path.exists(path), "part2_comparison.csv should exist."

    def test_part2_comparison_columns(self):
        """Comparison CSV has model and accuracy columns."""
        path = os.path.join(OUTPUT_DIR, "part2_comparison.csv")
        if not os.path.exists(path):
            pytest.skip("part2_comparison.csv not found")

        df = pd.read_csv(path)
        required = {"model", "accuracy"}
        assert required.issubset(set(df.columns)), (
            f"Comparison should have columns {required}, got {set(df.columns)}"
        )

    def test_part2_cnn_beats_dense(self):
        """CNN accuracy is higher than Dense accuracy."""
        path = os.path.join(OUTPUT_DIR, "part2_comparison.csv")
        if not os.path.exists(path):
            pytest.skip("part2_comparison.csv not found")

        df = pd.read_csv(path)
        dense_acc = df[df["model"] == "Dense"]["accuracy"].values
        cnn_acc = df[df["model"] == "CNN"]["accuracy"].values
        if len(dense_acc) == 0 or len(cnn_acc) == 0:
            pytest.skip("Comparison CSV missing Dense or CNN row")

        assert cnn_acc[0] > dense_acc[0], (
            f"CNN ({cnn_acc[0]:.2%}) should beat Dense ({dense_acc[0]:.2%})"
        )

    def test_part2_training_history_exists(self):
        """Training history plot was created and has content."""
        path = os.path.join(OUTPUT_DIR, "part2_training_history.png")
        assert os.path.exists(path), "part2_training_history.png should exist."
        size = os.path.getsize(path)
        assert size > 1000, (
            f"Training history plot seems too small ({size} bytes), may be corrupted"
        )


class TestPart3:
    """Test Part 3: LSTM on ECG5000"""

    def test_part3_output_exists(self):
        """part3_results.json was created."""
        path = os.path.join(OUTPUT_DIR, "part3_results.json")
        assert os.path.exists(path), (
            "part3_results.json should exist. Run the assignment notebook first."
        )

    def test_part3_has_required_keys(self):
        """part3_results.json has accuracy and confusion_matrix."""
        path = os.path.join(OUTPUT_DIR, "part3_results.json")
        if not os.path.exists(path):
            pytest.skip("part3_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert "accuracy" in results, "Results should contain 'accuracy'"
        assert "confusion_matrix" in results, "Results should contain 'confusion_matrix'"

    def test_part3_accuracy_threshold(self):
        """LSTM achieves >85% accuracy on ECG5000."""
        path = os.path.join(OUTPUT_DIR, "part3_results.json")
        if not os.path.exists(path):
            pytest.skip("part3_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert results["accuracy"] > 0.85, (
            f"LSTM should achieve >85% accuracy on ECG5000, got {results['accuracy']:.2%}"
        )

    def test_part3_confusion_matrix_shape(self):
        """Confusion matrix is 5x5 for 5-class classification."""
        path = os.path.join(OUTPUT_DIR, "part3_results.json")
        if not os.path.exists(path):
            pytest.skip("part3_results.json not found")

        with open(path) as f:
            results = json.load(f)

        cm = np.array(results["confusion_matrix"])
        assert cm.shape == (5, 5), f"Confusion matrix should be 5x5, got {cm.shape}"

    def test_part3_training_history_exists(self):
        """Training history plot was created and has content."""
        path = os.path.join(OUTPUT_DIR, "part3_training_history.png")
        assert os.path.exists(path), "part3_training_history.png should exist."
        size = os.path.getsize(path)
        assert size > 1000, (
            f"Training history plot seems too small ({size} bytes), may be corrupted"
        )
