---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Part 2b: Add Validation

Load and process patient data with BMI calculations.

**Your task:** Add schema and bounds validation to catch data quality issues early.

---

## Load data

```python
import pandas as pd

# TODO: Define a function validate_schema(df, required_columns) that:
#       - Checks if all required columns are present
#       - Raises ValueError with list of missing columns if any are missing

# TODO: Define a function validate_bounds(df, bounds_dict) that:
#       - For each column in bounds_dict, check if values are within (min, max)
#       - Use df[col].between(min, max) to find out-of-bounds values
#       - Raises ValueError showing patient_id and value for any out-of-bounds rows

df = pd.read_csv("data/patient_intake.csv")

# TODO: Call validate_schema() to check for required columns:
#       ["patient_id", "weight_kg", "height_cm", "age"]

# TODO: Call validate_bounds() with bounds:
#       weight_kg: (30, 250)
#       height_cm: (120, 230)
#       age: (0, 110)

df.head()
```

---

## Calculate BMI

```python
df["height_m"] = df["height_cm"] / 100
df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
df["bmi"] = df["bmi"].round(1)

df[["patient_id", "weight_kg", "height_cm", "bmi"]].head()
```

---

## Categorize BMI

```python
df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, float("inf")],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
    right=False
)

df[["patient_id", "bmi", "bmi_category"]].head()
```

---

## Summary statistics

```python
summary = df.groupby("bmi_category")["patient_id"].count()
print("\nBMI category distribution:")
print(summary)

high_risk = df[df["bmi"] > 30]
print(f"\nHigh-risk patients (BMI > 30): {len(high_risk)}")
```
