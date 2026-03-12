#!/usr/bin/env python3
"""
Demo 3a: Buggy BMI Calculator for VS Code Debugging Practice

This script has THREE intentional bugs to find using the VS Code debugger:
1. Formula error in calculate_bmi()
2. NameError in get_bmi_category()
3. IndexError in print_recommendations()

Practice debugging workflow:
- Set breakpoints and inspect variables
- Use Step Into/Over to trace execution
- Fix each bug and verify the fix works
"""


def calculate_bmi(weight, height):
    """Calculate BMI from weight (kg) and height (m)."""
    # BUG 1: Incorrect formula (missing square on height)
    # Correct formula: weight / (height ** 2)
    bmi = weight / height
    return bmi


def get_bmi_category(bmi):
    """Return the BMI category for a given BMI value."""
    # BUG 2: Typo in variable name causes NameError
    if bmi < 18.5:
        category = "Underweight" 
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return catgory  # NameError: 'catgory' is not defined


def print_recommendations(category_ids):
    """Print health recommendations based on BMI categories."""
    recommendations = [
        "Consider consulting a nutritionist",
        "Keep up the healthy habits",
        "Focus on balanced diet and exercise",
        "Consult healthcare provider for weight management",
    ]

    # BUG 3: Off-by-one error causes IndexError
    # Loop starts at 1 (skips first) and tries to access i+1 (out of bounds)
    print("\nRecommendations:")
    for i in range(1, len(category_ids)):
        print(f"{i}. {recommendations[i+1]}")


def main():
    """Main function to calculate and display BMI information."""
    print("=" * 50)
    print("BMI Calculator - VS Code Debugging Demo")
    print("=" * 50)
    # breakpoint() # Opens pdb, gutter bullet for VS Code debugging

    # Test with sample data instead of user input for reproducible debugging
    test_patients = [
        ("Alice", 70, 1.75),  # Should be normal weight
        ("Bob", 90, 1.80),    # Should be overweight
        ("Carol", 55, 1.60),  # Should be normal weight
    ]

    print("\nProcessing patients...")
    for name, weight, height in test_patients:
        print(f"\nPatient: {name}")
        print(f"Weight: {weight} kg, Height: {height} m")

        # BUG 1 will cause wrong BMI values
        bmi = calculate_bmi(weight, height)
        print(f"BMI: {bmi:.1f}")

        # BUG 2 will cause NameError here
        category = get_bmi_category(bmi)
        print(f"Category: {category}")

    # BUG 3 will cause IndexError here
    print_recommendations([1, 2, 3, 4])

    print("\n" + "=" * 50)
    print("Analysis complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
