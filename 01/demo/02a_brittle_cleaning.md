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

# Demo 2a: Brittle Patient Data Cleaning (BEFORE defensive programming)

This notebook loads patient intake data and calculates BMI. It works fine with clean data but **fails silently or cryptically** when data is messy.

**Your task:** Run this notebook with different data files and observe failures, then add defensive programming guardrails.

---

## Load and inspect data

```python
import pandas as pd

# Hardcoded path - breaks if file moves or doesn't exist
df = pd.read_csv("data/patient_intake.csv")

# No validation - assumes columns exist
df.head()
```

---

## Calculate BMI

```python
# No type checking - fails if values are strings
# No bounds checking - allows impossible values
df["height_m"] = df["height_cm"] / 100
df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
df["bmi"] = df["bmi"].round(1)

df[["patient_id", "weight_kg", "height_cm", "bmi"]].head()
```

---

## Categorize BMI

```python
# No handling for missing values
df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, float("inf")],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
    right=False
)

df[["patient_id", "bmi", "bmi_category"]].head(10)
```

---

## Filter and summarize

```python
# Silent failure if filtering removes all rows
high_bmi = df[df["bmi"] > 30]
print(f"Patients with BMI > 30: {len(high_bmi)}")

# No logging - can't trace what happened during analysis
summary = df.groupby("bmi_category")["patient_id"].count()
print("\nBMI category distribution:")
print(summary)
```

---

## What breaks with this notebook?

**Try running with these data files and observe the failures:**

1. **`patient_intake_missing_height.csv`** - Missing `height_cm` column
   - Error: `KeyError: 'height_cm'`
   - Problem: No schema validation before processing

2. **`patient_intake_bad_values.csv`** - Out-of-bounds values (500 kg weight, 50 cm height)
   - Silent failure: BMI calculations produce nonsense
   - Problem: No bounds checking

3. **Non-existent file:** Change path to `data/wrong_file.csv`
   - Error: `FileNotFoundError` with cryptic message
   - Problem: No existence check, no helpful error

**Other issues:**
- No logging means you can't trace execution or debug production issues
- If you add print/logging later, be careful not to expose PHI (patient IDs, names, etc.)

---

## Your task (Demo 2b will show the solution)

Add defensive programming to this notebook:

1. **Config instead of hardcoded paths:** Load file path and bounds from `02_config.yaml`
2. **Schema validation:** Check required columns exist before processing
3. **Bounds checking:** Validate weight, height, age are in realistic ranges
4. **Logging:** Add `logging.info()` to trace execution
5. **Graceful error handling:** Raise informative exceptions with context

Run the hardened version with all three data files—it should fail fast with clear error messages instead of silently producing wrong results.
