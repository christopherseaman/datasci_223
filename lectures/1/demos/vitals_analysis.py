#!/usr/bin/env python3

"""
Vital Signs Analysis
A simple script to analyze patient vital signs data
"""

def is_hypertensive(systolic, diastolic):
    """Determine if blood pressure readings indicate hypertension"""
    return systolic >= 130 or diastolic >= 80

def calculate_pulse_pressure(systolic, diastolic):
    """Calculate pulse pressure (difference between systolic and diastolic)"""
    return systolic - diastolic

# Sample patient data - in practice, would be loaded from CSV/database
patients = [
    {"id": "P1", "name": "Patient 1", "systolic": 142, "diastolic": 88, "pulse": 72},
    {"id": "P2", "name": "Patient 2", "systolic": 124, "diastolic": 77, "pulse": 68},
    {"id": "P3", "name": "Patient 3", "systolic": 158, "diastolic": 94, "pulse": 88},
    {"id": "P4", "name": "Patient 4", "systolic": 120, "diastolic": 70, "pulse": 65},
]

# Analyze each patient
hypertensive_count = 0

print("Vital Signs Analysis Report")
print("-" * 50)

for patient in patients:
    # Extract data
    patient_id = patient["id"]
    systolic = patient["systolic"]
    diastolic = patient["diastolic"]
    
    # Perform analysis
    hypertension_status = is_hypertensive(systolic, diastolic)
    pulse_pressure = calculate_pulse_pressure(systolic, diastolic)
    
    # Update statistics
    if hypertension_status:
        hypertensive_count += 1
    
    # Print results
    print(f"Patient {patient_id}:")
    print(f"  BP: {systolic}/{diastolic} mmHg")
    print(f"  Pulse Pressure: {pulse_pressure} mmHg")
    print(f"  Hypertension: {'Yes' if hypertension_status else 'No'}")
    print()

# Summary statistics
print("Summary:")
print(f"  Total patients: {len(patients)}")
print(f"  Hypertensive patients: {hypertensive_count} ({hypertensive_count/len(patients)*100:.1f}%)")
