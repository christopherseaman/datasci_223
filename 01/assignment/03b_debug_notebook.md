---
jupyter:
    jupytext:
        text_representation:
            extension: .md
            format_name: markdown
            format_version: "1.3"
    kernelspec:
        display_name: Python 3
        language: python
        name: python3
---

# Assignment Part 3b: Debug Lab Results Analysis

This notebook analyzes patient glucose test results to identify diabetes risk. It contains hidden issues you need to uncover using VS Code's notebook debugger.

**Your task:**

- Use the debug icon to run cells interactively
- Set breakpoints and inspect variables
- Fix the issues and add concise comments explaining each change
- Restart the kernel + Run All to verify everything works

---

## Setup and load data

```python
import pandas as pd
from pathlib import Path

# Load patient data
data_path = Path("data/patient_intake.csv")
patients = pd.read_csv(data_path)

print(f"Loaded {len(patients)} patients")
patients.head()
```

---

## Calculate fasting glucose estimates

```python
print("Estimating fasting glucose from BMI and age...")

# Simple glucose estimation for demonstration
# (In reality, would come from lab tests)
patients["glucose_mg_dl"] = (patients["weight_kg"] * 1.2 + patients["age"] * 0.3).round(0)

# Convert to string for "display formatting"
patients["glucose_mg_dl"] = patients["glucose_mg_dl"].astype(str)

print(f"Glucose calculated for {len(patients)} patients")
print(f"Sample values:")
print(patients[["patient_id", "glucose_mg_dl"]].head())
```

---

## Categorize diabetes risk

```python
print("\nCategorizing diabetes risk based on fasting glucose...")

def categorize_glucose(glucose_value):
    """Categorize diabetes risk from fasting glucose (mg/dL)."""
    if glucose_value < 100:
        return "Low risk (normal)"
    elif glucose_value < 126:
        return "High risk (prediabetes)"
    else:
        return "Very high risk (diabetes)"

patients["diabetes_risk"] = patients["glucose_mg_dl"].apply(categorize_glucose)

print("Risk categories assigned:")
print(patients[["patient_id", "glucose_mg_dl", "diabetes_risk"]].head(10))
```

---

## Filter high-risk patients

```python
print("\nIdentifying patients needing follow-up...")

# Find patients with elevated glucose
high_risk = patients[
    patients["diabetes_risk"].str.contains("High risk")
].copy()

print(f"Found {len(high_risk)} patients with elevated glucose")
if len(high_risk) > 0:
    print(f"Glucose range: {high_risk['glucose_mg_dl'].min()} to {high_risk['glucose_mg_dl'].max()}")
```

---

## Calculate intervention priority scores

```python
print("\nCalculating intervention priority scores...")

priority_patients = []
records = high_risk.to_dict("records")

# Calculate priority score for each high-risk patient
for i in range(1, len(records) + 1):
    patient = records[i]

    # Priority score: higher glucose + older age = higher priority
    glucose = float(patient["glucose_mg_dl"])
    age = int(patient["age"])
    priority_score = (glucose - 100) + (age * 0.5)

    priority_patients.append({
        "patient_id": patient["patient_id"],
        "glucose": glucose,
        "age": age,
        "priority_score": round(priority_score, 1)
    })

# Sort by priority score
priority_patients.sort(key=lambda x: x["priority_score"], reverse=True)

print(f"Priority scores calculated for {len(priority_patients)} patients")
if priority_patients:
    print(f"\nTop 3 priority patients:")
    for p in priority_patients[:3]:
        print(f"  {p['patient_id']}: score {p['priority_score']} (glucose={p['glucose']}, age={p['age']})")
```

---

## Summary

```python
print("\n" + "=" * 50)
print("Lab Results Analysis Complete")
print("=" * 50)
print(f"Total patients analyzed: {len(patients)}")
print(f"High-risk patients: {len(high_risk)}")
print(f"Patients prioritized for intervention: {len(priority_patients)}")
```

---

## Debugging Checklist

After fixing all bugs, verify:

- [ ] Runs without errors end-to-end
- [ ] Glucose values are reasonable numbers
- [ ] Risk categories make sense (high glucose = high risk)
- [ ] All high-risk patients are identified
- [ ] Priority scores calculated correctly
- [ ] Restart kernel + Run All completes successfully
- [ ] Added comments explaining each fix
