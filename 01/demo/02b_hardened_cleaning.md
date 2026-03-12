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

# Demo 2b: Hardened Patient Data Cleaning (AFTER defensive programming)

This notebook demonstrates defensive programming patterns:
- Configuration-driven paths and bounds
- Schema validation before processing
- Bounds checking with informative errors
- Logging to trace execution
- Pure functions for testability

**Compare to 02a_brittle_cleaning.md** to see what changed.

---

## Setup: Load config and initialize logging

```python
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

# Load configuration from YAML instead of hardcoding
DEMO_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CONFIG_PATH = DEMO_DIR / "02_config.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Set up logging with config-driven level
logging.basicConfig(
    level=getattr(logging, CONFIG["logging"]["level"]),
    format=CONFIG["logging"]["format"]
)

logging.info("Configuration loaded from %s", CONFIG_PATH.name)
logging.info("Demo directory: %s", DEMO_DIR)
```

---

## Helper functions: validate schema and bounds

```python
def load_patient_data(csv_path: Path, required_cols: list[str]) -> pd.DataFrame:
    """
    Load CSV and validate required columns exist.

    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If required columns missing
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Check the path in {CONFIG_PATH.name}"
        )

    df = pd.read_csv(csv_path)
    logging.info("Loaded %d rows from %s", len(df), csv_path.name)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df[required_cols].copy()


def validate_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """
    Validate numeric columns are within realistic bounds.

    Raises:
        ValueError: If values out of bounds, with details of bad rows
    """
    df = df.copy()

    # Convert to numeric and check for non-numeric values
    numeric_cols = list(bounds.keys())
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if df[numeric_cols].isna().any().any():
        bad_rows = df[df[numeric_cols].isna().any(axis=1)]
        raise ValueError(
            f"Non-numeric values found in {numeric_cols}\n"
            f"Problem rows:\n{bad_rows[['patient_id'] + numeric_cols]}"
        )

    # Check bounds for each column
    for col, limits in bounds.items():
        out_of_bounds = ~df[col].between(limits["min"], limits["max"])
        if out_of_bounds.any():
            bad_rows = df.loc[out_of_bounds, ["patient_id", col]]
            raise ValueError(
                f"Column '{col}' has values outside [{limits['min']}, {limits['max']}]\n"
                f"Problem rows:\n{bad_rows}"
            )

    logging.info("Bounds validation passed for %s", numeric_cols)
    return df


def calculate_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure function: calculate BMI and category.

    Doesn't modify input df, returns new df with added columns.
    """
    df = df.copy()
    df["height_m"] = df["height_cm"] / 100
    df["bmi"] = (df["weight_kg"] / (df["height_m"] ** 2)).round(1)

    # Use config thresholds for categorization
    thresholds = CONFIG["bmi_thresholds"]
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, thresholds["underweight"], thresholds["normal"],
              thresholds["overweight"], float("inf")],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False
    )

    logging.info("Calculated BMI for %d patients", len(df))
    return df
```

---

## Happy path: clean data processing

```python
# Load file path from config
data_path = DEMO_DIR / CONFIG["data"]["input_file"]
logging.info("Processing: %s", data_path.name)

# Validate schema and bounds before any calculations
intake_df = load_patient_data(data_path, CONFIG["required_columns"])
validated_df = validate_bounds(intake_df, CONFIG["bounds"])

# Calculate BMI using pure function
results_df = calculate_bmi(validated_df)

# Display results
results_df[["patient_id", "age", "sex", "bmi", "bmi_category"]].head(10)
```

---

## Test failure modes: observe informative errors

```python
# Test with messy data files
test_files = [
    "data/patient_intake_missing_height.csv",  # Missing column
    "data/patient_intake_bad_values.csv",      # Out of bounds
]

for test_file in test_files:
    test_path = DEMO_DIR / test_file
    if not test_path.exists():
        logging.warning("Test file not found: %s", test_file)
        continue

    logging.info("\n--- Testing with %s ---", test_file)
    try:
        df = load_patient_data(test_path, CONFIG["required_columns"])
        df = validate_bounds(df, CONFIG["bounds"])
        _ = calculate_bmi(df)
        logging.info("SUCCESS: %s passed all checks", test_file)
    except (FileNotFoundError, ValueError) as err:
        logging.error("EXPECTED FAILURE: %s", err)
```

---

## Jupyter magics for analysis

```python
# Jupyter magics help with quick checks and profiling
# Use them directly in notebooks (won't work in .py scripts)

# Where am I?
%pwd

# How long does BMI calculation take?
%timeit calculate_bmi(validated_df).head()

# Check file sizes
!ls -lh data/

# Display current variables
%whos DataFrame
```

---

## Summary statistics

```python
# Only runs if happy path succeeded
if "results_df" in locals():
    summary = results_df.groupby("bmi_category").agg({
        "patient_id": "count",
        "bmi": ["mean", "min", "max"]
    }).round(1)

    print("\nBMI Category Summary:")
    print(summary)

    # Log high-risk patients
    high_bmi_count = (results_df["bmi"] > 30).sum()
    logging.info("Patients with BMI > 30: %d (%.1f%%)",
                 high_bmi_count, 100 * high_bmi_count / len(results_df))
```

---

## Key defensive programming takeaways

1. **Configuration over hardcoding:** Paths, thresholds, and settings in YAML
2. **Fail fast with context:** Validate inputs before processing, raise informative errors
3. **Logging for observability:** Trace execution, especially for long-running analyses
4. **Pure functions:** No side effects = easier testing and debugging
5. **Type hints:** Document expected inputs/outputs for functions
6. **PHI safety:** Never log patient identifiers (patient_id, MRN, names, DOB)

### PHI Safety Note

Notice our logging statements avoid patient identifiers:
- ✅ `logging.info("Loaded %d rows", len(df))` - aggregate count only
- ✅ `logging.info("Bounds validation passed for %s", numeric_cols)` - column names
- ❌ `logging.info("Processing patient %s", patient_id)` - **NEVER do this**

**Error messages trade-off:** In this demo, validation errors show `patient_id` to help identify problem rows during development. In production with real PHI:
- Use row numbers instead: `f"Problem rows at indices: {bad_indices}"`
- Or log to secure audit trail with appropriate access controls
- Never include PHI in logs that go to stdout or general log files

**What changed from 02a?**
- ❌ `df = pd.read_csv("data/patient_intake.csv")`
- ✅ `df = load_patient_data(CONFIG["data"]["input_file"], CONFIG["required_columns"])`

- ❌ Silent calculation failures with bad data
- ✅ Explicit bounds validation with informative error messages

- ❌ No logging—can't trace what happened
- ✅ Logging at key points shows execution flow

Try breaking this notebook—it should fail with helpful errors, not silent corruption!
