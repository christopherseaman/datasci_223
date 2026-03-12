"""Tests for Assignment 8: Murder Mystery Agents.

Tests check output artifacts only — students run the notebook first,
then these tests verify the saved results.
"""

import json
import os
import re
import pytest

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")


# ---------- Part 1: Murder at the Mountain Cabin ----------

class TestPart1:
    @pytest.fixture(autouse=True)
    def load_results(self):
        path = os.path.join(OUTPUT_DIR, "part1_results.json")
        if os.path.exists(path):
            with open(path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_part1_results_exist(self):
        assert self.results is not None, "output/part1_results.json not found — run the notebook first"

    def test_part1_has_required_fields(self):
        assert self.results is not None, "output/part1_results.json not found"
        for field in ["killer", "weapon", "motive", "evidence"]:
            assert field in self.results, f"Missing field: {field}"

    def test_part1_killer_is_larry(self):
        assert self.results is not None, "output/part1_results.json not found"
        assert "larry" in self.results["killer"].lower(), (
            f"Expected killer to be Larry, got: {self.results['killer']}"
        )

    def test_part1_evidence_nonempty(self):
        assert self.results is not None, "output/part1_results.json not found"
        assert isinstance(self.results["evidence"], list), "evidence should be a list"
        assert len(self.results["evidence"]) > 0, "evidence list is empty"


# ---------- Part 2: Death at St. Mercy Hospital ----------

class TestPart2:
    @pytest.fixture(autouse=True)
    def load_results(self):
        path = os.path.join(OUTPUT_DIR, "part2_results.json")
        if os.path.exists(path):
            with open(path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_part2_results_exist(self):
        assert self.results is not None, "output/part2_results.json not found — run the notebook first"

    def test_part2_has_required_fields(self):
        assert self.results is not None, "output/part2_results.json not found"
        for field in ["killer", "weapon", "time_of_death", "reasoning"]:
            assert field in self.results, f"Missing field: {field}"

    def test_part2_killer_is_blake(self):
        assert self.results is not None, "output/part2_results.json not found"
        assert "blake" in self.results["killer"].lower(), (
            f"Expected killer to be Dr. Blake, got: {self.results['killer']}"
        )

    def test_part2_weapon_is_syringe(self):
        assert self.results is not None, "output/part2_results.json not found"
        assert "syringe" in self.results["weapon"].lower(), (
            f"Expected weapon to be syringe, got: {self.results['weapon']}"
        )

    def test_part2_time_of_death(self):
        """Accept 9:30, 09:30, 21:30, or 9.30 in any surrounding text."""
        assert self.results is not None, "output/part2_results.json not found"
        tod = self.results["time_of_death"]
        assert re.search(r"(^|[^\d])0?9[:.]30|21[:.]30", tod), (
            f"Expected time of death around 9:30, got: {tod}"
        )

    def test_part2_reasoning_substantive(self):
        assert self.results is not None, "output/part2_results.json not found"
        assert len(self.results["reasoning"]) >= 50, (
            f"Reasoning is only {len(self.results['reasoning'])} chars — "
            "provide detailed logical reasoning"
        )
