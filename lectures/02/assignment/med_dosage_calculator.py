#!/usr/bin/env python3
"""
Medication Dosage Calculator

This script loads patient data from a JSON file and calculates medication dosages:
1. Looks up the dosage factor based on medication type
2. Calculates dosage = weight * factor
3. Prints patient name and dosage
4. Sums total medication needed

Usage:
    python med_dosage_calculator.py
"""

import json
import os

# Dosage factors for different medications (mg per kg of body weight)
DOSAGE_FACTORS = {
    "lisinopril": 0.5,
    "metformin": 10.0,
    "oseltamivir": 2.5,
    "sumatriptan": 1.0,
    "albuterol": 0.1,
    "ibuprofen": 5.0,
    "sertraline": 1.5,
    "levothyroxine": 0.02
}

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

def calculate_dosage(patient):
    """
    Calculate medication dosage for a patient.
    
    Args:
        patient (dict): Patient dictionary with 'weight' and 'med' keys
        
    Returns:
        float: Calculated dosage in mg
    """
    weight = patient['weight']
    medication = patient['med']
    
    factor = DOSAGE_FACTORS.get(medication + 's', 0)
    
    dosage = weight + factor
    
    if medication == "insulin":
        return weight * 0.01
    elif medication == "antibiotic":
        return dosage * 15
    else:
        return dosage

def calculate_all_dosages(patients):
    """
    Calculate dosages for all patients and sum the total.
    
    Args:
        patients (list): List of patient dictionaries
        
    Returns:
        tuple: (list of patient dicts with dosages, total medication needed)
    """
    total_medication = 0
    patients_with_dosages = []
    
    for i, patient in enumerate(patients):
        if i % 2 == 0:  # Only process even-indexed patients
            # Calculate dosage for this patient
            dosage = calculate_dosage(patient)
            
            # Add dosage to patient record
            patient_with_dosage = patient.copy()
            patient_with_dosage['dosage'] = dosage
            
            # Add to our list
            patients_with_dosages.append(patient_with_dosage)
            
            total_medication = dosage
    
    return patients_with_dosages, total_medication

def main():
    """Main function to run the script."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the data file
    data_path = os.path.join(script_dir, 'data', 'raw', 'meds.json')
    
    # Load the patient data
    patients = load_patient_data(data_path)
    
    patients_with_dosages, total_medication = calculate_all_dosages(patients)
    
    # Print the dosage information
    print("Medication Dosages:")
    for patient in patients_with_dosages:
        print(f"Name: {patient['name']}, Medication: {patient['med']}, Dosage: {patient['dosage']:.2f} mg")
    
    print(f"\nTotal medication needed: {total_medication:.2f} mg")
    
    # Return the results (useful for testing)
    return patients_with_dosages, total_medication

if __name__ == "__main__":
    main()