---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Demo 3b: Buggy Patient Analysis Notebook (VS Code Notebook Debugging)

This notebook has subtle bugs that require using VS Code's notebook debugger to find. Practice:
- Clicking the **debug icon** next to a cell
- Setting breakpoints inside notebook cells
- Using Variables panel to inspect DataFrames
- Stepping through code to find logic errors

**Known bugs:**
1. Type mismatch when filtering by age (string vs int comparison)
2. Off-by-one error in loop iteration
3. Logic error in BMI risk categorization

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

## BUG 1: Filter elderly patients (age comparison issue)

```python
# This cell has a BUG: ages might be stored as strings
# Try debugging: click the debug icon on this cell, set a breakpoint,
# and inspect patients["age"].dtype in the Variables panel

print("Finding patients over 65...")

# BUG: If age is string type, this comparison doesn't work as expected
# "70" > 65 evaluates differently than int comparison
elderly = patients[patients["age"] > 65].copy()

print(f"Found {len(elderly)} elderly patients")

# If this is unexpectedly 0, the bug is likely a type issue
if len(elderly) == 0:
    print("⚠️ WARNING: No elderly patients found - check data types!")
```

---

## Fix for BUG 1 (uncomment after finding the issue)

```python
# Ensure age is numeric
# patients["age"] = pd.to_numeric(patients["age"], errors="coerce")
# elderly = patients[patients["age"] > 65].copy()
# print(f"After type fix: {len(elderly)} elderly patients")
```

---

## BUG 2: Calculate summary statistics with off-by-one error

```python
# This cell has a BUG in the loop range
# Set a breakpoint inside the loop and watch the 'i' variable

print("\nPatient summaries:")
patient_ids = patients["patient_id"].tolist()

# BUG: range(1, len(patient_ids)) skips first patient and may exceed bounds
# Should be range(len(patient_ids))
for i in range(1, len(patient_ids)):
    row = patients.iloc[i]
    # When i equals len(patients)-1, this tries to access iloc[len(patients)]
    next_row = patients.iloc[i + 1]  # IndexError on last iteration

    print(f"Patient {row['patient_id']}: BMI trend analysis")
```

---

## Fix for BUG 2 (uncomment after finding the issue)

```python
# print("\nCorrected patient summaries:")
# for i in range(len(patient_ids)):
#     row = patients.iloc[i]
#     print(f"Patient {row['patient_id']}: Age {row['age']}, BMI analysis")
```

---

## BUG 3: Categorize BMI risk (logic error)

```python
# This cell has a LOGIC BUG in the risk categorization
# Debug: Set breakpoint in the function, step through with different BMI values

def categorize_risk(bmi):
    """Categorize health risk based on BMI."""
    # BUG: Logic is reversed - high BMI should be high risk, not low
    if bmi < 18.5:
        return "High risk"
    elif bmi < 25:
        return "Moderate risk"
    elif bmi < 30:
        return "Low risk"
    else:
        return "Very low risk"  # Should be "Very high risk"!

# Test the function
test_cases = [17, 22, 28, 35]
print("\nRisk categorization test:")
for bmi in test_cases:
    risk = categorize_risk(bmi)
    print(f"BMI {bmi}: {risk}")

# BUG: BMI of 35 should be "Very high risk", not "Very low risk"!
```

---

## Fix for BUG 3 (uncomment after finding the issue)

```python
# def categorize_risk_fixed(bmi):
#     """Corrected risk categorization."""
#     if bmi < 18.5:
#         return "Moderate risk (underweight)"
#     elif bmi < 25:
#         return "Low risk (normal)"
#     elif bmi < 30:
#         return "Moderate risk (overweight)"
#     else:
#         return "High risk (obese)"
#
# print("\nCorrected risk categorization:")
# for bmi in test_cases:
#     risk = categorize_risk_fixed(bmi)
#     print(f"BMI {bmi}: {risk}")
```

---

## Debugging tips for notebooks

1. **Use the debug icon:** Click the debug button beside a cell (looks like a bug icon)
2. **Set breakpoints:** Click in the gutter beside code lines inside cells
3. **Variables panel:** Inspect DataFrames, check `.dtype`, `.shape`, `.head()`
4. **Step through:** Use Step Over (F10) to execute line-by-line
5. **Restart kernel:** After fixing bugs, restart kernel and Run All to verify clean state

---

## Summary: What you should have found

- **BUG 1:** Age column is string type, comparison `"70" > 65` behaves unexpectedly
  - **Fix:** Convert to numeric with `pd.to_numeric()`

- **BUG 2:** Loop starts at 1 (skips first patient) and accesses `i+1` (IndexError)
  - **Fix:** Use `range(len(...))` and remove the `+1` offset

- **BUG 3:** Risk logic is inverted—high BMI labeled as "Very low risk"
  - **Fix:** Reverse the risk labels or fix the conditional logic

**Key lesson:** Notebook debugging is just like script debugging, but you debug individual cells. Always restart kernel + Run All after fixes to ensure clean state!
