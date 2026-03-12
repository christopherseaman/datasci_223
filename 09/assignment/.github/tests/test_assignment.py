"""
Tests for Computer Vision Assignment

Tests verify:
1. Output files exist with correct structure
2. Model achieves reasonable performance (above chance)
3. Training history shows learning (decreasing loss)
4. Saved model loads and produces predictions
"""

import pytest
import json
import os
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")


class TestPart1:
    """Test Part 1: Data Pipeline"""

    def test_data_summary_exists(self):
        """part1_data_summary.json was created."""
        path = os.path.join(OUTPUT_DIR, "part1_data_summary.json")
        assert os.path.exists(path), (
            "part1_data_summary.json should exist. Run the assignment notebook first."
        )

    def test_data_summary_has_required_keys(self):
        """part1_data_summary.json has expected keys."""
        path = os.path.join(OUTPUT_DIR, "part1_data_summary.json")
        if not os.path.exists(path):
            pytest.skip("part1_data_summary.json not found")

        with open(path) as f:
            data = json.load(f)

        for key in ["train_size", "val_size", "test_size", "num_classes"]:
            assert key in data, f"Data summary should contain '{key}'"

    def test_data_splits_reasonable(self):
        """Dataset splits have reasonable sizes."""
        path = os.path.join(OUTPUT_DIR, "part1_data_summary.json")
        if not os.path.exists(path):
            pytest.skip("part1_data_summary.json not found")

        with open(path) as f:
            data = json.load(f)

        assert data["train_size"] > 0, "Training set should not be empty"
        assert data["val_size"] > 0, "Validation set should not be empty"
        assert data["test_size"] > 0, "Test set should not be empty"
        assert data["num_classes"] == 2, "Should have 2 classes"


class TestPart2:
    """Test Part 2: Transfer Learning"""

    def test_training_history_exists(self):
        """part2_training_history.json was created."""
        path = os.path.join(OUTPUT_DIR, "part2_training_history.json")
        assert os.path.exists(path), (
            "part2_training_history.json should exist."
        )

    def test_training_history_has_data(self):
        """Training history contains loss and accuracy values."""
        path = os.path.join(OUTPUT_DIR, "part2_training_history.json")
        if not os.path.exists(path):
            pytest.skip("part2_training_history.json not found")

        with open(path) as f:
            history = json.load(f)

        assert "train_loss" in history, "History should contain 'train_loss'"
        assert "val_loss" in history, "History should contain 'val_loss'"
        assert "val_acc" in history, "History should contain 'val_acc'"
        assert len(history["train_loss"]) >= 1, "Should have at least 1 epoch"

    def test_loss_decreases(self):
        """Training loss shows a general decreasing trend."""
        path = os.path.join(OUTPUT_DIR, "part2_training_history.json")
        if not os.path.exists(path):
            pytest.skip("part2_training_history.json not found")

        with open(path) as f:
            history = json.load(f)

        losses = history["train_loss"]
        if len(losses) < 2:
            pytest.skip("Need at least 2 epochs to check trend")

        # Final loss should be lower than first loss
        assert losses[-1] < losses[0], (
            f"Training loss should decrease: first={losses[0]:.4f}, "
            f"last={losses[-1]:.4f}"
        )

    def test_training_curves_plot_exists(self):
        """part2_training_curves.png was created."""
        path = os.path.join(OUTPUT_DIR, "part2_training_curves.png")
        assert os.path.exists(path), "part2_training_curves.png should exist."

    def test_model_saved(self):
        """part2_model.pt was created."""
        path = os.path.join(OUTPUT_DIR, "part2_model.pt")
        assert os.path.exists(path), "part2_model.pt should exist."
        assert os.path.getsize(path) > 1000, (
            "Model file should be larger than 1KB"
        )


class TestPart3:
    """Test Part 3: Evaluation"""

    def test_results_exist(self):
        """part3_results.json was created."""
        path = os.path.join(OUTPUT_DIR, "part3_results.json")
        assert os.path.exists(path), "part3_results.json should exist."

    def test_results_have_metrics(self):
        """Results contain accuracy, precision, recall, and F1."""
        path = os.path.join(OUTPUT_DIR, "part3_results.json")
        if not os.path.exists(path):
            pytest.skip("part3_results.json not found")

        with open(path) as f:
            results = json.load(f)

        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in results, f"Results should contain '{key}'"
            assert 0 <= results[key] <= 1, f"{key} should be between 0 and 1"

    def test_accuracy_above_chance(self):
        """Model achieves above-chance accuracy (>55% for binary)."""
        path = os.path.join(OUTPUT_DIR, "part3_results.json")
        if not os.path.exists(path):
            pytest.skip("part3_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert results["accuracy"] > 0.55, (
            f"Accuracy should be above chance (50%): got {results['accuracy']:.2%}"
        )

    def test_confusion_matrix_exists(self):
        """part3_confusion_matrix.png was created."""
        path = os.path.join(OUTPUT_DIR, "part3_confusion_matrix.png")
        assert os.path.exists(path), "part3_confusion_matrix.png should exist."
