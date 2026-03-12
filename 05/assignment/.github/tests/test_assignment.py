"""
Tests for Classification Assignment (Fashion-MNIST)

Tests verify:
1. Output files exist with correct structure
2. Models achieve reasonable performance
3. CV results have expected format and reasonable values
"""

import pytest
import json
import os
import pandas as pd
import numpy as np

# Output directory path
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")


class TestPart1:
    """Test Part 1: Binary Classification (T-shirt vs Trouser)"""

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
        """Part 1 achieves >95% accuracy (this is an easy binary task)."""
        path = os.path.join(OUTPUT_DIR, "part1_results.json")
        if not os.path.exists(path):
            pytest.skip("part1_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert results["accuracy"] > 0.95, (
            f"Part 1 should achieve >95% accuracy on T-shirt vs Trouser, got {results['accuracy']:.2%}"
        )

    def test_part1_confusion_matrix_shape(self):
        """Confusion matrix is 2x2 for binary classification."""
        path = os.path.join(OUTPUT_DIR, "part1_results.json")
        if not os.path.exists(path):
            pytest.skip("part1_results.json not found")

        with open(path) as f:
            results = json.load(f)

        cm = np.array(results["confusion_matrix"])
        assert cm.shape == (2, 2), f"Confusion matrix should be 2x2, got {cm.shape}"

    def test_part1_confusion_matrix_reasonable(self):
        """Confusion matrix values are reasonable (not all zeros, diagonal dominant)."""
        path = os.path.join(OUTPUT_DIR, "part1_results.json")
        if not os.path.exists(path):
            pytest.skip("part1_results.json not found")

        with open(path) as f:
            results = json.load(f)

        cm = np.array(results["confusion_matrix"])
        assert cm.sum() > 0, "Confusion matrix should not be all zeros"
        # Diagonal should be larger than off-diagonal for a good classifier
        assert cm[0, 0] > cm[0, 1], "More TN than FP expected"
        assert cm[1, 1] > cm[1, 0], "More TP than FN expected"


class TestPart2:
    """Test Part 2: Multi-class Classification with CV (Footwear types)"""

    def test_part2_cv_results_exists(self):
        """part2_cv_results.csv was created."""
        path = os.path.join(OUTPUT_DIR, "part2_cv_results.csv")
        assert os.path.exists(path), (
            "part2_cv_results.csv should exist. Run the assignment notebook first."
        )

    def test_part2_cv_results_columns(self):
        """CV results has model, fold, score columns."""
        path = os.path.join(OUTPUT_DIR, "part2_cv_results.csv")
        if not os.path.exists(path):
            pytest.skip("part2_cv_results.csv not found")

        df = pd.read_csv(path)
        required = {"model", "fold", "score"}
        assert required.issubset(set(df.columns)), (
            f"CV results should have columns {required}, got {set(df.columns)}"
        )

    def test_part2_both_models_compared(self):
        """Both LogisticRegression and RandomForest were compared."""
        path = os.path.join(OUTPUT_DIR, "part2_cv_results.csv")
        if not os.path.exists(path):
            pytest.skip("part2_cv_results.csv not found")

        df = pd.read_csv(path)
        models = set(df["model"].unique())

        assert len(models) >= 2, f"Should compare at least 2 models, found {len(models)}"

    def test_part2_five_folds(self):
        """5-fold CV was used for each model."""
        path = os.path.join(OUTPUT_DIR, "part2_cv_results.csv")
        if not os.path.exists(path):
            pytest.skip("part2_cv_results.csv not found")

        df = pd.read_csv(path)
        folds_per_model = df.groupby("model")["fold"].nunique()

        for model, n_folds in folds_per_model.items():
            assert n_folds >= 5, f"{model} should have 5 folds, got {n_folds}"

    def test_part2_cv_scores_reasonable(self):
        """CV scores are in valid range and reasonable for this task."""
        path = os.path.join(OUTPUT_DIR, "part2_cv_results.csv")
        if not os.path.exists(path):
            pytest.skip("part2_cv_results.csv not found")

        df = pd.read_csv(path)
        scores = df["score"]

        assert scores.min() >= 0.0, "Scores should be >= 0"
        assert scores.max() <= 1.0, "Scores should be <= 1"
        assert scores.mean() > 0.80, (
            f"Mean CV accuracy should be >80% for footwear classification, got {scores.mean():.2%}"
        )

    def test_part2_test_results_exists(self):
        """part2_test_results.csv was created."""
        path = os.path.join(OUTPUT_DIR, "part2_test_results.csv")
        assert os.path.exists(path), "part2_test_results.csv should exist."

    def test_part2_test_results_has_metrics(self):
        """Test results contains classification metrics (precision, recall, f1)."""
        path = os.path.join(OUTPUT_DIR, "part2_test_results.csv")
        if not os.path.exists(path):
            pytest.skip("part2_test_results.csv not found")

        df = pd.read_csv(path, index_col=0)
        # classification_report DataFrame should have these columns
        expected_metrics = {"precision", "recall", "f1-score"}
        assert expected_metrics.issubset(set(df.columns)), (
            f"Test results should have metrics {expected_metrics}, got {set(df.columns)}"
        )


class TestPart3:
    """Test Part 3: Full Model Comparison Pipeline (Clothing vs Footwear)"""

    def test_part3_cv_results_exists(self):
        """part3_cv_results.csv was created."""
        path = os.path.join(OUTPUT_DIR, "part3_cv_results.csv")
        assert os.path.exists(path), (
            "part3_cv_results.csv should exist. Run the assignment notebook first."
        )

    def test_part3_three_models_compared(self):
        """All 3 models (LogisticRegression, RandomForest, XGBoost) were compared."""
        path = os.path.join(OUTPUT_DIR, "part3_cv_results.csv")
        if not os.path.exists(path):
            pytest.skip("part3_cv_results.csv not found")

        df = pd.read_csv(path)
        models = set(df["model"].unique())

        assert len(models) >= 3, f"Should compare 3 models, found {len(models)}"

    def test_part3_cv_uses_auc(self):
        """CV results use AUC metric (not accuracy)."""
        path = os.path.join(OUTPUT_DIR, "part3_cv_results.csv")
        if not os.path.exists(path):
            pytest.skip("part3_cv_results.csv not found")

        df = pd.read_csv(path)
        assert "auc" in df.columns, "CV results should have 'auc' column"

    def test_part3_cv_auc_reasonable(self):
        """CV AUC scores are high for this easy task."""
        path = os.path.join(OUTPUT_DIR, "part3_cv_results.csv")
        if not os.path.exists(path):
            pytest.skip("part3_cv_results.csv not found")

        df = pd.read_csv(path)
        mean_auc = df["auc"].mean()

        assert mean_auc > 0.95, (
            f"Mean CV AUC should be >0.95 for Clothing vs Footwear, got {mean_auc:.3f}"
        )

    def test_part3_roc_curves_exists(self):
        """ROC curves plot was created."""
        path = os.path.join(OUTPUT_DIR, "part3_roc_curves.png")
        assert os.path.exists(path), "part3_roc_curves.png should exist."

    def test_part3_roc_curves_not_empty(self):
        """ROC curves file has content (not empty)."""
        path = os.path.join(OUTPUT_DIR, "part3_roc_curves.png")
        if not os.path.exists(path):
            pytest.skip("part3_roc_curves.png not found")

        size = os.path.getsize(path)
        assert size > 1000, f"ROC curves file seems too small ({size} bytes), may be corrupted"

    def test_part3_confusion_matrix_exists(self):
        """Confusion matrix plot was created."""
        path = os.path.join(OUTPUT_DIR, "part3_confusion_matrix.png")
        assert os.path.exists(path), "part3_confusion_matrix.png should exist."

    def test_part3_test_results_exists(self):
        """part3_test_results.csv was created."""
        path = os.path.join(OUTPUT_DIR, "part3_test_results.csv")
        assert os.path.exists(path), "part3_test_results.csv should exist."

    def test_part3_test_results_columns(self):
        """Test results has model, accuracy, auc columns."""
        path = os.path.join(OUTPUT_DIR, "part3_test_results.csv")
        if not os.path.exists(path):
            pytest.skip("part3_test_results.csv not found")

        df = pd.read_csv(path)
        required = {"model", "accuracy", "auc"}
        assert required.issubset(set(df.columns)), (
            f"Test results should have columns {required}, got {set(df.columns)}"
        )

    def test_part3_test_auc_threshold(self):
        """Final model achieves >0.95 AUC on test set."""
        path = os.path.join(OUTPUT_DIR, "part3_test_results.csv")
        if not os.path.exists(path):
            pytest.skip("part3_test_results.csv not found")

        df = pd.read_csv(path)
        auc = df["auc"].iloc[0]

        assert auc > 0.95, (
            f"Part 3 should achieve >0.95 AUC on Clothing vs Footwear, got {auc:.3f}"
        )

    def test_part3_test_accuracy_threshold(self):
        """Final model achieves >90% accuracy on test set."""
        path = os.path.join(OUTPUT_DIR, "part3_test_results.csv")
        if not os.path.exists(path):
            pytest.skip("part3_test_results.csv not found")

        df = pd.read_csv(path)
        accuracy = df["accuracy"].iloc[0]

        assert accuracy > 0.90, (
            f"Part 3 should achieve >90% accuracy on Clothing vs Footwear, got {accuracy:.2%}"
        )
