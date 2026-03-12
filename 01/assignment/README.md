# Assignment 01: Reliable Notebooks and Debugging

**Due:** Before Lecture 02
**Points:** Pass/Fail (autograded)
**Skills:** Environment setup, defensive programming, VS Code debugging

## Overview

This assignment has **three parts** that build on Lecture 01 concepts:

1. **Email Verification** - Verify environment setup (1 script)
2. **Defensive Programming** - Add logging, validation, config to notebooks (3 notebooks)
3. **Debugging** - Fix bugs in script and notebook (1 script + 1 notebook)

**All work is autograded** via GitHub Actions - push frequently to see test results!

## Assignment Structure

```
assignment/
├── 01_process_email.py              # Part 1: Email verification
├── 02a_logging.ipynb                # Part 2a: Add logging
├── 02b_validation.ipynb             # Part 2b: Add validation
├── 02c_config_driven.ipynb          # Part 2c: Use config file
├── 03a_debug_script.py              # Part 3a: Debug Python script
├── 03b_debug_notebook.ipynb         # Part 3b: Debug notebook
├── config.yaml                      # Config for 02c
└── data/                            # Test data files
```

## Part 1: Email Verification

**File:** `01_process_email.py`

Verify your UCSF email and that your environment works.

### Tasks

1. Run the script with your UCSF email:

    ```bash
    python3 01_process_email.py your.email@ucsf.edu
    ```

2. Verify output:

    ```bash
    cat processed_email.txt  # Should show a 64-character hash
    ```

3. Commit and push:
    ```bash
    git add processed_email.txt
    git commit -m "feat: add email verification"
    git push
    ```

### Success Criteria

- [ ] `processed_email.txt` exists with valid SHA256 hash
- [ ] Hash matches course roster

## Part 2: Defensive Programming

Convert brittle notebooks into robust, production-ready code through three progressive steps.

### Part 2a: Add Logging

**File:** `02a_logging.ipynb`

**Current state:** Notebook runs but provides no visibility into execution.

**Your task:** Add logging to trace execution and help debug issues.

**Requirements:**

1. Import and configure logging at the top:

    ```python
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    ```

2. Add logging statements at key points:
    - After loading data: `logging.info("Loaded %d patients", len(df))`
    - Before calculations: `logging.info("Calculating BMI...")`
    - After filtering: `logging.info("Found %d high-risk patients", count)`

3. Run notebook - output should show execution trace

**Test:** All logging tests pass in pytest

### Part 2b: Add Validation

**File:** `02b_validation.ipynb`

**Current state:** Notebook crashes cryptically with bad data or produces wrong results silently.

**Your task:** Add schema and bounds validation with helpful error messages.

**Requirements:**

1. **Schema validation** - Check required columns exist:

    ```python
    required_cols = ["patient_id", "weight_kg", "height_cm", "age"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    ```

2. **Bounds validation** - Check realistic value ranges:

    ```python
    bounds = {
        "weight_kg": (30, 250),
        "height_cm": (120, 230),
        "age": (0, 110)
    }

    for col, (min_val, max_val) in bounds.items():
        out_of_bounds = ~df[col].between(min_val, max_val)
        if out_of_bounds.any():
            bad_rows = df[out_of_bounds][["patient_id", col]]
            raise ValueError(f"{col} out of bounds:\n{bad_rows}")
    ```

3. Test with bad data files (provided) - should fail with clear errors

**Test:** Run notebook with `data/patient_intake_bad_values.csv` - should raise helpful ValueError

### Part 2c: Config-Driven Development

**File:** `02c_config_driven.ipynb`

**Current state:** File paths and thresholds hardcoded throughout notebook.

**Your task:** Move all configuration to `config.yaml` file.

**Requirements:**

1. Create/update `config.yaml`:

    ```yaml
    data:
        input_file: "data/patient_intake.csv"

    bounds:
        weight_kg: { min: 30, max: 250 }
        height_cm: { min: 120, max: 230 }
        age: { min: 0, max: 110 }

    bmi_thresholds:
        underweight: 18.5
        normal: 25
        overweight: 30 # obese is any value above this threshold
    ```

2. Load config in notebook:

    ```python
    import yaml
    from pathlib import Path

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Use config values
    data_path = Path(config["data"]["input_file"])
    bounds = config["bounds"]
    ```

3. Replace ALL hardcoded values with config lookups

**Test:** Change config values, rerun notebook - should use new values without code changes

## Part 3: Debugging with VS Code

Practice systematic debugging for both scripts and notebooks.

### Part 3a: Debug Python Script

**File:** `03a_debug_script.py`

**Current state:** Script has THREE intentional bugs.

**Your task:** Use VS Code debugger to find and fix all bugs, documenting each fix.

**Known bugs:**

1. **Formula error** - BMI calculation missing exponent
2. **NameError** - Return statement references `risk_level` even though the variable is named `risk_lvl`
3. **Loop bug** - `range(len(patients) - 1)` skips the final patient and leaves results incomplete

**Debugging workflow:**

1. Set breakpoint on line with bug
2. Run debugger (Debug icon → Python File)
3. When paused:
    - Check **Variables panel** for unexpected values
    - Add **Watch expression** to compare correct calculation
    - Use **Debug Console** to test fixes
4. Fix bug and add comment:
    ```python
    # BUG 1: Used weight/height instead of weight/(height**2)
    # FIX: Added exponent for correct BMI formula
    bmi = weight / (height ** 2)
    ```

**Test:** Script runs without errors and produces correct BMI values

### Part 3b: Debug Notebook

**File:** `03b_debug_notebook.ipynb`

**Current state:** The lab-results notebook estimates fasting glucose from intake data and assigns diabetes risk, but hidden issues in data typing, filtering, and scoring produce incorrect counts.

**Your task:** Use VS Code’s notebook debugger to uncover each issue, add concise `FIX:` comments beside your changes, then restart the kernel and Run All to confirm clean execution.

**Investigation clues:**

1. Glucose values get reformatted for display—watch how that affects comparisons in the risk categorization cell.
2. The follow-up filter should capture both “High risk” and “Very high risk” labels; step through the string matching logic while paused.
3. The intervention priority loop should compute a score for every high-risk patient exactly once—inspect the index math as you iterate.

**Debugging workflow:**

1. Click the **debug icon** beside a cell (bug/play button).
2. Set breakpoints inside the problematic cell.
3. When paused:
    - **Variables panel:** Inspect `patients["glucose_mg_dl"]` and `patients["age"]` dtypes/values.
    - **Watch panel:** Track expressions like `len(high_risk)` or `records[i]`.
    - **Debug Console:** Run snippets such as `patients.head()` or `high_risk[:3]`.
4. Apply fixes, document them with comments, restart the kernel, and Run All to verify.

**Test:** Restart kernel + Run All completes without errors

## Testing Your Work

```bash
pip install -r requirements.txt
pytest .github/tests/test_assignment.py -v
```

## Submission Checklist

- [ ] `processed_email.txt` committed (Part 1)
- [ ] `02a_logging.ipynb` has logging statements
- [ ] `02b_validation.ipynb` has schema + bounds checks
- [ ] `02c_config_driven.ipynb` loads from `config.yaml`
- [ ] `03a_debug_script.py` all bugs fixed with comments
- [ ] `03b_debug_notebook.ipynb` all bugs fixed with comments
- [ ] All files pushed to GitHub
- [ ] GitHub Actions tests pass 🤞

## Grading

**Pass/Fail** If autograding fails, then a human will review

| Part | Tests Must Pass |
| - | |
| **1. Email** | Valid hash file exists |
| **2a. Logging** | Notebook executes, logging output present |
| **2b. Validation** | Notebook runs with clean data, catches bad data |
| **2c. Config** | Notebook uses config.yaml, modifying config changes behavior |
| **3a. Debug Script** | Script runs without errors, BMI calculations correct |
| **3b. Debug Notebook** | Notebook executes cleanly, produces expected output |

## Getting Help

### Stuck?

1. Review demos in `lectures/01/demo/` - they show these patterns
2. Read test output in GitHub Actions - tells you what's wrong
3. Ask in lab!

### Tips:

- **Part 2:** Look at `02b_hardened_cleaning.ipynb` demo for patterns
- **Part 3a:** Similar bugs to `03a_buggy_bmi.py` demo
- **Part 3b:** Review `03b_buggy_analysis.ipynb` demo for notebook debugging
- **Git:** Commit after each part, push frequently to see test results
