#!/usr/bin/env python3

def calculate_bmi(weight, height):
    """Calculate BMI from weight (kg) and height (m)."""
    # Bug 1: Incorrect BMI formula (missing square)
    bmi = weight / height  # Should be weight / (height ** 2)
    return bmi

def get_bmi_category(bmi):
    """Return the BMI category for a given BMI value."""
    # Bug 2: Typo in variable name
    if bmi < 18.5:
        catgory = "Underweight"  # Should be 'category'
    elif bmi < 25:
        catgory = "Normal weight"
    elif bmi < 30:
        catgory = "Overweight"
    else:
        catgory = "Obese"
    return category

def print_recommendations(numbers):
    """Print health recommendations based on BMI category."""
    recommendations = [
        "Consider consulting a nutritionist",
        "Keep up the healthy habits",
        "Focus on balanced diet and exercise",
        "Consult healthcare provider"
    ]
    
    # Bug 3: Off-by-one error in loop and indexing
    # Skips first element and will cause IndexError on last iteration
    for i in range(1, len(numbers)):
        print(f"Recommendation {i}: {recommendations[i+1]}")

def main():
    """Main function to calculate and display BMI information."""
    print("BMI Calculator")
    print("-------------")
    
    # Get user input
    weight = float(input("Enter weight (kg): "))
    height = float(input("Enter height (m): "))
    
    # Calculate BMI
    bmi = calculate_bmi(weight, height)
    print(f"\nYour BMI is: {bmi:.1f}")
    
    # Get and print category
    category = get_bmi_category(bmi)
    print(f"Category: {category}")
    
    # Print recommendations
    print("\nRecommendations:")
    print_recommendations([1, 2, 3, 4])

if __name__ == "__main__":
    main() 