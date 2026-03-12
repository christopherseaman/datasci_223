"""
Tests for Assignment 7: Clinical NLP with LLMs and Embeddings

Tests verify output artifacts only — students run the notebook first,
then these tests check the saved results.
"""

import pytest
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")


class TestPart1:
    """Test Part 1: Clinical Entity Extraction output."""

    def test_extraction_results_exist(self):
        """extraction_results.json was created."""
        path = os.path.join(OUTPUT_DIR, "extraction_results.json")
        assert os.path.exists(path), (
            "extraction_results.json should exist in output/. Run the assignment notebook first."
        )

    def test_extraction_results_is_list(self):
        """extraction_results.json contains a JSON list."""
        path = os.path.join(OUTPUT_DIR, "extraction_results.json")
        if not os.path.exists(path):
            pytest.skip("extraction_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert isinstance(results, list), "extraction_results.json should contain a list"

    def test_extraction_results_nonempty(self):
        """At least one extraction succeeded."""
        path = os.path.join(OUTPUT_DIR, "extraction_results.json")
        if not os.path.exists(path):
            pytest.skip("extraction_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert len(results) >= 1, (
            "At least one extraction should succeed (got 0 results)"
        )

    def test_extraction_result_structure(self):
        """Each result has diagnosis, medications, lab_values, and confidence."""
        path = os.path.join(OUTPUT_DIR, "extraction_results.json")
        if not os.path.exists(path):
            pytest.skip("extraction_results.json not found")

        with open(path) as f:
            results = json.load(f)

        if len(results) == 0:
            pytest.skip("No extraction results to check")

        required = {"diagnosis", "medications", "lab_values", "confidence"}
        for i, result in enumerate(results):
            assert isinstance(result, dict), f"Result {i} should be a dict"
            missing = required - set(result.keys())
            assert not missing, (
                f"Result {i} missing required fields: {missing}"
            )

    def test_extraction_confidence_range(self):
        """Confidence values are between 0 and 1."""
        path = os.path.join(OUTPUT_DIR, "extraction_results.json")
        if not os.path.exists(path):
            pytest.skip("extraction_results.json not found")

        with open(path) as f:
            results = json.load(f)

        for i, result in enumerate(results):
            conf = result.get("confidence")
            if conf is not None:
                assert 0.0 <= conf <= 1.0, (
                    f"Result {i} confidence {conf} should be between 0 and 1"
                )


class TestPart2:
    """Test Part 2: Semantic Search output."""

    def test_search_results_exist(self):
        """search_results.json was created."""
        path = os.path.join(OUTPUT_DIR, "search_results.json")
        assert os.path.exists(path), (
            "search_results.json should exist in output/. Run the assignment notebook first."
        )

    def test_search_results_is_list(self):
        """search_results.json contains a JSON list."""
        path = os.path.join(OUTPUT_DIR, "search_results.json")
        if not os.path.exists(path):
            pytest.skip("search_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert isinstance(results, list), "search_results.json should contain a list"

    def test_search_results_count(self):
        """Search returned the expected number of results."""
        path = os.path.join(OUTPUT_DIR, "search_results.json")
        if not os.path.exists(path):
            pytest.skip("search_results.json not found")

        with open(path) as f:
            results = json.load(f)

        assert len(results) == 3, (
            f"Expected 3 search results (top_k=3), got {len(results)}"
        )

    def test_search_result_structure(self):
        """Each result has 'note' and 'score' keys."""
        path = os.path.join(OUTPUT_DIR, "search_results.json")
        if not os.path.exists(path):
            pytest.skip("search_results.json not found")

        with open(path) as f:
            results = json.load(f)

        for i, result in enumerate(results):
            assert isinstance(result, dict), f"Result {i} should be a dict"
            assert "note" in result, f"Result {i} missing 'note' key"
            assert "score" in result, f"Result {i} missing 'score' key"

    def test_search_scores_are_floats(self):
        """Scores are numeric values."""
        path = os.path.join(OUTPUT_DIR, "search_results.json")
        if not os.path.exists(path):
            pytest.skip("search_results.json not found")

        with open(path) as f:
            results = json.load(f)

        for i, result in enumerate(results):
            score = result.get("score")
            assert isinstance(score, (int, float)), (
                f"Result {i} score should be numeric, got {type(score).__name__}"
            )

    def test_search_scores_sorted_descending(self):
        """Results are sorted by score descending."""
        path = os.path.join(OUTPUT_DIR, "search_results.json")
        if not os.path.exists(path):
            pytest.skip("search_results.json not found")

        with open(path) as f:
            results = json.load(f)

        if len(results) < 2:
            pytest.skip("Need at least 2 results to check sorting")

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            "Results should be sorted by score descending"
        )

    def test_search_notes_are_strings(self):
        """Note values are non-empty strings."""
        path = os.path.join(OUTPUT_DIR, "search_results.json")
        if not os.path.exists(path):
            pytest.skip("search_results.json not found")

        with open(path) as f:
            results = json.load(f)

        for i, result in enumerate(results):
            note = result.get("note")
            assert isinstance(note, str), (
                f"Result {i} note should be a string"
            )
            assert len(note.strip()) > 0, (
                f"Result {i} note should not be empty"
            )
