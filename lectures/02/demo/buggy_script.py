def calculate_bmi(weight, height):
    # Intentional bug: wrong formula (should be weight / height**2)
    return weight / height

def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def main():
    patient_weight = 70  # kg
    patient_height = 1.75  # meters

    bmi = calculate_bmi(patient_weight, patient_height)
    print("BMI:", bmi)

    category = categorize_bmi(bmi)
    print("Category:", category)

    # Intentional bug: typo in variable name
    if catgory == "Obese":
        print("Recommend weight loss program.")

    # Intentional bug: off-by-one error
    numbers = [1, 2, 3, 4, 5]
    for i in range(1, len(numbers)): # Wrong range of indices
        print(numbers[i+1])  # IndexError on last iteration

if __name__ == "__main__":
    main()