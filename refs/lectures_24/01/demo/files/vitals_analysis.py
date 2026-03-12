def analyze_blood_pressure(systolic, diastolic):
    """Analyze blood pressure readings and return category."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif systolic < 140 or diastolic < 90:
        return "Stage 1 Hypertension"
    else:
        return "Stage 2 Hypertension"

def calculate_bmi(weight_kg, height_m):
    """Calculate Body Mass Index."""
    return weight_kg / (height_m ** 2)

def analyze_patient(age, blood_pressure, weight_kg, height_m):
    """Analyze patient vitals and return summary."""
    systolic, diastolic = map(int, blood_pressure.split('/'))
    bp_category = analyze_blood_pressure(systolic, diastolic)
    bmi = calculate_bmi(weight_kg, height_m)
    
    return {
        "age": age,
        "blood_pressure": blood_pressure,
        "bp_category": bp_category,
        "bmi": round(bmi, 1)
    }

if __name__ == "__main__":
    # Example patient data
    patient = analyze_patient(65, "140/90", 70, 1.75)
    
    print("Patient Analysis:")
    print(f"Age: {patient['age']}")
    print(f"Blood Pressure: {patient['blood_pressure']}")
    print(f"BP Category: {patient['bp_category']}")
    print(f"BMI: {patient['bmi']}")
    
    # Additional BMI examples
    print("\nBMI Calculations:")
    print(f"BMI for 70kg, 1.75m: {calculate_bmi(70, 1.75):.1f}")
    print(f"BMI for 80kg, 1.80m: {calculate_bmi(80, 1.80):.1f}") 