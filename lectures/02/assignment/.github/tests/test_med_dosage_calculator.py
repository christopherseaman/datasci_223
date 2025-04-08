#!/usr/bin/env python3
"""
Tests for the med_dosage_calculator.py script.

This script tests the functionality of the medication dosage calculator,
ensuring it correctly calculates dosages based on weight and medication type,
and correctly sums the total medication needed.

Usage:
    pytest test_med_dosage_calculator.py
"""

import os
import json
import pytest
import tempfile
from med_dosage_calculator import load_patient_data, calculate_dosage, calculate_all_dosages, DOSAGE_FACTORS

# Sample test data
TEST_DATA = [
    {
        "name": "john smith",
        "weight": 80.0,
        "med": "lisinopril",
        "frequency": "daily"
    },
    {
        "name": "sarah johnson",
        "weight": 60.0,
        "med": "oseltamivir",
        "frequency": "twice daily"
    },
    {
        "name": "robert williams",
        "weight": 90.0,
        "med": "metformin",
        "frequency": "daily"
    }
]

@pytest.fixture
def sample_data_file():
    """Create a temporary file with sample patient data."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(TEST_DATA, f)
        temp_file_name = f.name
    
    yield temp_file_name
    
    # Cleanup after test
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

def test_load_patient_data(sample_data_file):
    """Test that patient data is loaded correctly from a file."""
    patients = load_patient_data(sample_data_file)
    assert len(patients) == 3
    assert patients[0]["name"] == "john smith"
    assert patients[0]["weight"] == 80.0
    assert patients[0]["med"] == "lisinopril"

def test_calculate_dosage():
    """Test that dosage is calculated correctly for a single patient."""
    patient = TEST_DATA[0]  # John Smith, 80kg, lisinopril
    dosage = calculate_dosage(patient)
    
    # lisinopril factor is 0.5 mg/kg, so 80kg * 0.5 = 40mg
    expected_dosage = 80.0 * 0.5
    assert dosage == expected_dosage
    
    # Test another patient
    patient = TEST_DATA[1]  # Sarah Johnson, 60kg, oseltamivir
    dosage = calculate_dosage(patient)
    
    # oseltamivir factor is 2.5 mg/kg, so 60kg * 2.5 = 150mg
    expected_dosage = 60.0 * 2.5
    assert dosage == expected_dosage

def test_calculate_all_dosages():
    """Test that dosages are calculated correctly for all patients and total is correct."""
    patients_with_dosages, total_medication = calculate_all_dosages(TEST_DATA)
    
    # Check that all patients have dosages
    assert len(patients_with_dosages) == 3
    
    # Check individual dosages
    assert patients_with_dosages[0]["dosage"] == 80.0 * 0.5  # John: 80kg * 0.5 = 40mg
    assert patients_with_dosages[1]["dosage"] == 60.0 * 2.5  # Sarah: 60kg * 2.5 = 150mg
    assert patients_with_dosages[2]["dosage"] == 90.0 * 10.0  # Robert: 90kg * 10.0 = 900mg
    
    # Check total medication
    expected_total = (80.0 * 0.5) + (60.0 * 2.5) + (90.0 * 10.0)  # 40 + 150 + 900 = 1090mg
    assert total_medication == expected_total

def test_dosage_factors():
    """Test that the dosage factors dictionary contains expected values."""
    assert "lisinopril" in DOSAGE_FACTORS
    assert DOSAGE_FACTORS["lisinopril"] == 0.5
    assert DOSAGE_FACTORS["metformin"] == 10.0
    assert DOSAGE_FACTORS["oseltamivir"] == 2.5

if __name__ == "__main__":
    pytest.main(["-v", __file__])