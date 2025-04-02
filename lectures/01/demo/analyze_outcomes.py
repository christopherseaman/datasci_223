#!/usr/bin/env python3

# Simple health outcomes analysis script

def calculate_risk_score(age, systolic_bp, cholesterol):
    """Calculate cardiovascular risk score based on simple factors"""
    # This is a simplified example, not for clinical use
    base_risk = (age / 10) * 0.1
    bp_factor = (systolic_bp - 120) / 10 * 0.1 if systolic_bp > 120 else 0
    chol_factor = (cholesterol - 200) / 10 * 0.05 if cholesterol > 200 else 0
    return base_risk + bp_factor + chol_factor

# Example usage
if __name__ == "__main__":
    print("Health Risk Calculator")
    patient_data = [
        {"id": "P1001", "age": 65, "bp": 140, "chol": 210},
        {"id": "P1002", "age": 45, "bp": 120, "chol": 180},
    ]
    
    for patient in patient_data:
        risk = calculate_risk_score(patient["age"], patient["bp"], patient["chol"])
        print(f"Patient {patient['id']}: Risk score = {risk:.2f}")
