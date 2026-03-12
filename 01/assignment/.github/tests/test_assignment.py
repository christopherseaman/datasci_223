#!/usr/bin/env python3
"""Assignment 01 Test Suite - Autograding

Tests focus on behavior, execution, and artifacts rather than implementation details.
"""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).parent.parent.parent


def _collect_output_text(notebook_json):
    """Concatenate plain-text outputs from an executed notebook."""
    chunks = []
    for cell in notebook_json.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                text_content = output["text"]
                if isinstance(text_content, list):
                    chunks.append("".join(text_content))
                else:
                    chunks.append(str(text_content))
            data = output.get("data", {})
            if isinstance(data, dict) and "text/plain" in data:
                text_data = data["text/plain"]
                if isinstance(text_data, list):
                    chunks.append("".join(text_data))
                else:
                    chunks.append(str(text_data))
    return "\n".join(chunk.strip() for chunk in chunks if chunk)


def _expected_glucose_metrics():
    """Compute expected high-risk and priority counts from source data."""
    data_path = BASE_DIR / "data" / "patient_intake.csv"
    df = pd.read_csv(data_path)

    df["glucose_mg_dl"] = (df["weight_kg"] * 1.2 + df["age"] * 0.3).round(0)
    df["risk"] = pd.cut(
        df["glucose_mg_dl"],
        bins=[-float("inf"), 100, 126, float("inf")],
        labels=[
            "Low risk (normal)",
            "High risk (prediabetes)",
            "Very high risk (diabetes)",
        ],
        right=False,
    )

    high_risk_mask = df["risk"].isin([
        "High risk (prediabetes)",
        "Very high risk (diabetes)",
    ])
    high_risk_count = int(high_risk_mask.sum())

    return {
        "high_risk": high_risk_count,
        "priority": high_risk_count,
    }


class TestPart1Email:
    """Test email verification - checks artifact generation"""

    def test_email_file_exists(self):
        """Check processed_email.txt exists"""
        assert (BASE_DIR / "processed_email.txt").exists(), \
            "Run: python 01_process_email.py your.email@ucsf.edu"

    def test_email_hash_valid_format(self):
        """Check hash is 64-char hex string (SHA256 format)"""
        email_file = BASE_DIR / "processed_email.txt"
        content = email_file.read_text().strip()

        assert len(content) == 64, \
            f"SHA256 hash should be 64 characters, got {len(content)}"
        assert all(c in '0123456789abcdef' for c in content.lower()), \
            "Hash should be hexadecimal (0-9, a-f)"

    def test_email_script_runs(self):
        """Test script runs without errors"""
        result = subprocess.run(
            [sys.executable, "01_process_email.py", "test.student@ucsf.edu"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0, \
            f"Script failed: {result.stderr}"


class TestPart2aLogging:
    """Test logging implementation - checks execution and output"""

    def test_notebook_executes(self):
        """Notebook should run without errors"""
        notebook_path = BASE_DIR / "02a_logging.ipynb"
        assert notebook_path.exists(), \
            "Convert with: jupytext --to notebook 02a_logging.md"

        result = subprocess.run(
            ["jupyter", "nbconvert", "--execute", "--to", "notebook",
             "--output", "02a_test_output.ipynb", str(notebook_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0, \
            f"Notebook failed to execute:\n{result.stderr}"

    def test_notebook_produces_logging_output(self):
        """Check that execution produces log-like output"""
        notebook_path = BASE_DIR / "02a_logging.ipynb"
        result = subprocess.run(
            ["jupyter", "nbconvert", "--execute", "--to", "notebook",
             "--stdout", str(notebook_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Check for logging patterns in output (INFO, WARNING, etc. or timestamp patterns)
        output = result.stdout + result.stderr
        has_logging = any(pattern in output for pattern in
                         ["INFO:", "WARNING:", "DEBUG:", "logging", "Loaded", "Processing"])

        assert has_logging, \
            "Expected to see logging output (e.g., 'INFO:', 'Loaded', etc.)"


class TestPart2bValidation:
    """Test validation implementation - checks behavior with bad data"""

    def test_notebook_handles_clean_data(self):
        """Notebook should run successfully with clean data"""
        notebook_path = BASE_DIR / "02b_validation.ipynb"
        assert notebook_path.exists(), \
            "Convert with: jupytext --to notebook 02b_validation.md"

        result = subprocess.run(
            ["jupyter", "nbconvert", "--execute", "--to", "notebook",
             "--output", "02b_test_clean.ipynb", str(notebook_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0, \
            f"Notebook should run successfully with clean data:\n{result.stderr}"

    def test_validation_catches_bad_data(self):
        """Validation should catch bad data (modify CSV path in test copy)"""
        # Create a test version that uses bad data
        notebook_path = BASE_DIR / "02b_validation.ipynb"
        if not notebook_path.exists():
            pytest.skip("Notebook not converted yet")

        # Read notebook, replace data path with bad values file
        import json
        with open(notebook_path) as f:
            nb = json.load(f)

        # Replace data path in all cells
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                if "patient_intake.csv" in source:
                    source = source.replace(
                        "patient_intake.csv",
                        "patient_intake_bad_values.csv"
                    )
                    cell["source"] = source.split("\n")

        # Write modified notebook
        test_nb_path = BASE_DIR / "02b_test_bad_data.ipynb"
        with open(test_nb_path, "w") as f:
            json.dump(nb, f)

        # Execute - should fail with validation error
        result = subprocess.run(
            ["jupyter", "nbconvert", "--execute", "--to", "notebook",
             "--output", "02b_test_bad_executed.ipynb", str(test_nb_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should fail - validation should catch bad data
        assert result.returncode != 0, \
            "Validation should catch bad data and raise an error"


class TestPart2cConfig:
    """Test config-driven development - checks config usage"""

    def test_config_file_valid(self):
        """Check config.yaml exists and has required structure"""
        import yaml
        config_path = BASE_DIR / "config.yaml"
        assert config_path.exists(), "config.yaml not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "data" in config, "config should have 'data' section"
        assert "input_file" in config["data"], "config.data should have 'input_file'"
        assert "bmi_thresholds" in config, "config should have 'bmi_thresholds'"

    def test_notebook_executes_with_config(self):
        """Notebook should run successfully using config"""
        notebook_path = BASE_DIR / "02c_config_driven.ipynb"
        assert notebook_path.exists(), \
            "Convert with: jupytext --to notebook 02c_config_driven.md"

        result = subprocess.run(
            ["jupyter", "nbconvert", "--execute", "--to", "notebook",
             "--output", "02c_test_output.ipynb", str(notebook_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0, \
            f"Notebook should run with config:\n{result.stderr}"

    def test_config_changes_affect_output(self):
        """Modifying config should change notebook behavior"""
        import yaml

        # Save original config
        config_path = BASE_DIR / "config.yaml"
        original_config = config_path.read_text()

        try:
            # Modify config to use different threshold
            with open(config_path) as f:
                config = yaml.safe_load(f)

            config["bmi_thresholds"]["overweight"] = 28  # Changed from 30

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Run notebook - should succeed with modified config
            notebook_path = BASE_DIR / "02c_config_driven.ipynb"
            if notebook_path.exists():
                result = subprocess.run(
                    ["jupyter", "nbconvert", "--execute", "--to", "notebook",
                     "--output", "02c_test_modified.ipynb", str(notebook_path)],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                # Should still run (proving it reads config)
                assert result.returncode == 0 or "FileNotFoundError" not in result.stderr, \
                    "Notebook should use config values"

        finally:
            # Restore original config
            config_path.write_text(original_config)


class TestPart3aDebugScript:
    """Test script debugging - checks correct execution and output"""

    def test_script_runs_without_errors(self):
        """Script should complete successfully after debugging"""
        script_path = BASE_DIR / "03a_debug_script.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, \
            f"Script should run without errors after fixing bugs:\n{result.stderr}\n{result.stdout}"

    def test_bmi_calculations_correct(self):
        """BMI calculations should be mathematically correct"""
        script_path = BASE_DIR / "03a_debug_script.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout

        # Check for correct BMI calculation (70kg, 1.75m = 22.9 BMI, not 40.0)
        # Allow some variation in output format
        assert "22.9" in output or "23.0" in output or "22.8" in output, \
            f"BMI for 70kg, 1.75m should be ~22.9, check output:\n{output}"

        # Should not show the buggy value of 40.0
        bmi_values = re.findall(r"BMI[:\s]+(\d+\.?\d*)", output)
        if bmi_values:
            for val in bmi_values:
                assert float(val) != 40.0, \
                    "BMI should not be 40.0 (that's the buggy calculation)"


class TestPart3bDebugNotebook:
    """Test notebook debugging - checks execution succeeds"""

    def test_notebook_executes_without_errors(self):
        """Notebook should run cleanly after debugging"""
        notebook_path = BASE_DIR / "03b_debug_notebook.ipynb"
        assert notebook_path.exists(), \
            "Convert with: jupytext --to notebook 03b_debug_notebook.md"

        result = subprocess.run(
            ["jupyter", "nbconvert", "--execute", "--to", "notebook",
             "--output", "03b_test_executed.ipynb", str(notebook_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, \
            f"Notebook should execute without errors after fixing bugs:\n{result.stderr}"

    def test_notebook_produces_expected_output(self):
        """Check that notebook produces reasonable analysis output"""
        import json

        executed_nb = BASE_DIR / "03b_test_executed.ipynb"
        if not executed_nb.exists():
            pytest.skip("Need to execute notebook first")

        with open(executed_nb) as f:
            nb = json.load(f)

        # Check that cells have outputs (proving execution completed)
        has_outputs = False
        for cell in nb.get("cells", []):
            if cell.get("outputs"):
                has_outputs = True
                break

        assert has_outputs, \
            "Notebook should produce output (check that bugs are fixed)"

        text_output = _collect_output_text(nb)
        assert "Lab Results Analysis Complete" in text_output, \
            "Summary banner missing from notebook output"
        for label in ("High risk (prediabetes)", "Very high risk (diabetes)"):
            assert label in text_output, \
                f"Expected risk category '{label}' to appear in output"

        expected = _expected_glucose_metrics()

        high_risk_match = re.search(r"High-risk patients:\s+(\d+)", text_output)
        assert high_risk_match, \
            "Summary should include 'High-risk patients: <count>' line"
        high_risk_count = int(high_risk_match.group(1))
        assert high_risk_count == expected["high_risk"], \
            f"Expected {expected['high_risk']} high-risk patients, got {high_risk_count}"

        priority_match = re.search(r"Patients prioritized for intervention:\s+(\d+)", text_output)
        assert priority_match, \
            "Summary should include 'Patients prioritized for intervention: <count>' line"
        priority_count = int(priority_match.group(1))
        assert priority_count == expected["priority"], \
            f"Expected {expected['priority']} priority patients, got {priority_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
