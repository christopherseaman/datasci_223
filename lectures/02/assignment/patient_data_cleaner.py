#!/usr/bin/env python3
"""
Patient Data Cleaner

This script loads patient data from a JSON file, cleans it by:
1. Capitalizing patient names
2. Converting ages to integers
3. Filtering out patients under 18
4. Printing the cleaned list

Usage:
    python patient_data_cleaner.py
"""

import json
import os

def load_patient_data(filepath):
    """
    Load patient data from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        list: List of patient dictionaries
    """
    with open(filepath, 'r') as file:
        return json.load(file)

def clean_patient_data(patients):
    """
    Clean patient data by:
    - Capitalizing names
    - Converting ages to integers
    - Filtering out patients under 18
    
    Args:
        patients (list): List of patient dictionaries
        
    Returns:
        list: Cleaned list of patient dictionaries
    """
    cleaned_patients = []
    
    for patient in patients:
        # Capitalize name (BUG: typo in key 'nage' instead of 'name')
        patient['nage'] = patient['name'].title()
        
        # BUG: Wrong method name (fill_na vs fillna)
        patient['age'] = patient['age'].fill_na(0)
        
        # BUG: Wrong method name (drop_duplcates vs drop_duplicates)
        patient = patient.drop_duplcates()
        
        # BUG: Wrong comparison operator (= vs ==)
        if patient['age'] = 18:
            # BUG: Logic error - keeps patients under 18 instead of filtering them out
            cleaned_patients.append(patient)
    
    return cleaned_patients

def main():
    """Main function to run the script."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the data file
    data_path = os.path.join(script_dir, 'data', 'raw', 'patients.json')
    
    # Load the patient data
    patients = load_patient_data(data_path)
    
    # Clean the patient data
    cleaned_patients = clean_patient_data(patients)
    
    # Print the cleaned patient data
    print("Cleaned Patient Data:")
    for patient in cleaned_patients:
        print(f"Name: {patient['name']}, Age: {patient['age']}, Diagnosis: {patient['diagnosis']}")
    
    # Return the cleaned data (useful for testing)
    return cleaned_patients

if __name__ == "__main__":
    main()