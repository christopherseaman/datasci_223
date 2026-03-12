def calculate_medication_dosage(weight_kg, age_years, medication_type):
    """Calculate medication dosage based on patient weight and age."""
    # Bug 1: Wrong variable name (should be weight_kg, not weight)
    if weight_kg < 0 or age_years < 0:
        return "Invalid input parameters"
    
    # Bug 2: Incorrect calculation (should be weight_kg * 0.5, not weight_kg * 5)
    base_dosage = weight_kg * 0.5
    
    # Bug 3: Missing condition for age adjustment
    if age_years > 65:
        # Bug 4: Wrong adjustment factor (should be 0.75, not 0.5)
        base_dosage = base_dosage * 0.75
    
    # Bug 5: Incorrect medication type check (should be 'acetaminophen', not 'acetaminophin')
    if medication_type == 'acetaminophin':
        return base_dosage
    elif medication_type == 'ibuprofen':
        # Bug 6: Wrong calculation (should be base_dosage * 1.2, not base_dosage * 12)
        return base_dosage * 1.2
    else:
        return "Unknown medication type"

def main():
    # Test cases
    patient1_weight = 70  # kg
    patient1_age = 45     # years
    
    # Bug 7: Wrong function call (should be calculate_medication_dosage, not calculate_dosage)
    dosage1 = calculate_medication_dosage(patient1_weight, patient1_age, 'acetaminophen')
    print(f"Patient 1 dosage: {dosage1} mg")
    
    patient2_weight = 65  # kg
    patient2_age = 72     # years
    
    # Bug 8: Wrong parameter order (should be weight_kg, age_years, medication_type)
    dosage2 = calculate_medication_dosage(patient2_weight, patient2_age, 'ibuprofen')
    print(f"Patient 2 dosage: {dosage2} mg")
    
    # Bug 9: Off-by-one error in loop
    print("\nDosage table for patient 1:")
    for i in range(0, 5):  # Should be range(0, 5) to print 5 rows
        weight = patient1_weight + i
        dosage = calculate_medication_dosage(weight, patient1_age, 'acetaminophen')
        print(f"Weight: {weight} kg, Dosage: {dosage} mg")

if __name__ == "__main__":
    main() 