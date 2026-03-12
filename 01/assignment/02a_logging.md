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

# Part 2a: Add Logging

Load and process patient data with BMI calculations.

**Your task:** Add logging statements to track the data processing steps.

---

## Load data

```python
import pandas as pd

# TODO: Import logging module
# TODO: Configure logging with basicConfig (level=INFO, format="%(levelname)s:%(message)s")
# TODO: Add logging.info() statement before reading CSV

df = pd.read_csv("data/patient_intake.csv")

# TODO: Add logging.info() statement after loading (mention number of rows loaded)

df.head()
```

---

## Calculate BMI

```python
# TODO: Add logging.info() statement at start of BMI calculation

df["height_m"] = df["height_cm"] / 100
df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
df["bmi"] = df["bmi"].round(1)

# TODO: Add logging.info() statement after BMI calculation (mention BMI range)

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

# TODO: Add logging.info() statement summarizing processing completion
```
