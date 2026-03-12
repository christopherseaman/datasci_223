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

# Part 2c: Config-Driven Development

Load and process patient data with BMI calculations.

**Your task:** Load configuration from `config.yaml` instead of hardcoding values.

---

## Load configuration

```python
import pandas as pd
import yaml
from pathlib import Path

# TODO: Load config.yaml using yaml.safe_load()
# TODO: Store result in a variable called 'config'

# Example structure you'll get:
# config = {
#     "data": {"input_file": "data/patient_intake.csv"},
#     "bounds": {
#         "weight_kg": {"min": 30, "max": 250},
#         "height_cm": {"min": 120, "max": 230},
#         "age": {"min": 0, "max": 110}
#     },
#     "bmi_thresholds": {
#         "underweight": 18.5,
#         "normal": 25,
#         "overweight": 30
#     }
# }
```

---

## Load data

```python
# TODO: Replace hardcoded path with config["data"]["input_file"]
df = pd.read_csv("data/patient_intake.csv")

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
# TODO: Replace hardcoded thresholds with values from config["bmi_thresholds"]
#       Use: underweight, normal, overweight thresholds from config
#       Bins should be: [0, underweight, normal, overweight, inf]

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

# TODO: Replace hardcoded 30 with config["bmi_thresholds"]["overweight"]
high_risk = df[df["bmi"] > 30]
print(f"\nHigh-risk patients (BMI > 30): {len(high_risk)}")
```
