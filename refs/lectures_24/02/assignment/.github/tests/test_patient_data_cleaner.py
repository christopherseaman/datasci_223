#!/usr/bin/env python3
"""
Tests for the patient_data_cleaner.py script.

This script tests the functionality of the patient data cleaner,
ensuring it correctly capitalizes names, converts ages to integers,
and filters out patients under 18.

Usage:
    pytest test_patient_data_cleaner.py
"""

import os
import json
import pytest
import tempfile
from patient_data_cleaner import load_patient_data, clean_patient_data

# Sample test data
TEST_DATA = [
    {
        "name": "john smith",
        "age": "32",
        "gender": "male",
        "diagnosis": "hypertension"
    },
    {
        "name": "sarah johnson",
        "age": "17",
        "gender": "female",
        "diagnosis": "influenza"
    },
    {
        "name": "robert williams",
        "age": "45",
        "gender": "male",
        "diagnosis": "diabetes"
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
    assert patients[1]["age"] == "17"

def test_clean_patient_data():
    """Test that patient data is cleaned correctly."""
    cleaned = clean_patient_data(TEST_DATA)
    
    # Should only have 2 patients (filtered out the 17-year-old)
    assert len(cleaned) == 2
    
    # Names should be capitalized
    assert cleaned[0]["name"] == "John Smith"
    assert cleaned[1]["name"] == "Robert Williams"
    
    # Ages should be integers
    assert isinstance(cleaned[0]["age"], int)
    assert cleaned[0]["age"] == 32
    assert cleaned[1]["age"] == 45
    
    # Check that the 17-year-old was filtered out
    ages = [p["age"] for p in cleaned]
    assert 17 not in ages

if __name__ == "__main__":
    pytest.main(["-v", __file__])