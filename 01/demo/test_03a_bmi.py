"""
Test suite for 03a_buggy_bmi.py

Demonstrates how to write tests to lock in bug fixes.
This is the same pattern used in assignment autograding.

IMPORTANT: These tests will FAIL until you fix the bugs in 03a_buggy_bmi.py!

Workflow:
1. Run pytest - tests fail (shows bugs exist)
2. Use VS Code debugger to fix bugs in 03a_buggy_bmi.py
3. Run pytest again - tests pass (confirms fixes work)

Run with: pytest test_03a_bmi.py -v
"""

import pytest
from pathlib import Path
import sys

# Add demo directory to path so we can import the BMI module
demo_dir = Path(__file__).parent
sys.path.insert(0, str(demo_dir))

# Import the module (Python doesn't like module names starting with numbers)
# So we use importlib to load it
import importlib.util

spec = importlib.util.spec_from_file_location("bmi_module", demo_dir / "03a_buggy_bmi.py")
bmi_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bmi_module)

# Now we can use the functions
calculate_bmi = bmi_module.calculate_bmi
get_bmi_category = bmi_module.get_bmi_category


class TestCalculateBMI:
    """Test BMI calculation formula."""

    def test_normal_weight_calculation(self):
        """Test BMI calculation with typical values."""
        bmi = calculate_bmi(70, 1.75)  # 70 kg, 1.75 m
        assert 22 < bmi < 23, f"Expected BMI ~22.9, got {bmi}"

    def test_bmi_bounds(self):
        """Test BMI stays within realistic bounds."""
        bmi = calculate_bmi(80, 1.80)
        assert 15 < bmi < 50, f"BMI {bmi} is outside realistic range"

    def test_underweight_calculation(self):
        """Test BMI for underweight example."""
        bmi = calculate_bmi(50, 1.70)
        assert bmi < 18.5, f"Expected underweight BMI, got {bmi}"

    def test_overweight_calculation(self):
        """Test BMI for overweight example."""
        bmi = calculate_bmi(90, 1.75)
        assert 25 < bmi < 30, f"Expected overweight BMI, got {bmi}"


class TestGetBMICategory:
    """Test BMI categorization logic."""

    def test_underweight_category(self):
        """Test underweight classification."""
        category = get_bmi_category(17.0)
        assert category == "Underweight"

    def test_normal_weight_category(self):
        """Test normal weight classification."""
        category = get_bmi_category(22.0)
        assert category == "Normal weight"

    def test_overweight_category(self):
        """Test overweight classification."""
        category = get_bmi_category(27.0)
        assert category == "Overweight"

    def test_obese_category(self):
        """Test obese classification."""
        category = get_bmi_category(32.0)
        assert category == "Obese"

    def test_boundary_values(self):
        """Test category boundaries are correct."""
        assert get_bmi_category(18.4) == "Underweight"
        assert get_bmi_category(18.5) == "Normal weight"
        assert get_bmi_category(24.9) == "Normal weight"
        assert get_bmi_category(25.0) == "Overweight"
        assert get_bmi_category(29.9) == "Overweight"
        assert get_bmi_category(30.0) == "Obese"


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_complete_workflow(self):
        """Test full BMI calculation and categorization workflow."""
        # Calculate BMI
        bmi = calculate_bmi(70, 1.75)

        # Categorize
        category = get_bmi_category(bmi)

        # Verify reasonable output
        assert category in ["Underweight", "Normal weight", "Overweight", "Obese"]
        assert category == "Normal weight", f"70kg at 1.75m should be normal weight, got {category}"


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
