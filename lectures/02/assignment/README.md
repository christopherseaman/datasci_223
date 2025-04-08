# Assignment: Debugging and Big Data Analysis 🐛📊

---

## Overview

This assignment has two parts:

1. **Debugging Python code (70% of grade)**
2. **Analyzing large health data (30% of grade)**

---

## Part 1: Debugging (70%)

### Tasks

- Fix the provided buggy scripts:
  - `patient_data_cleaner_buggy.py` (cleans and filters patient records)
  - `med_dosage_calculator_buggy.py` (calculates medication dosages)
- The bugs are **typical beginner mistakes** such as:
  - Typos and wrong variable names (`NameError`)
  - Off-by-one errors (`IndexError`)
  - Incorrect logic or formulas
  - Syntax issues (`SyntaxError`, `IndentationError`)
- The code intent is **simple and easy to reason about** (e.g., cleaning patient data, calculating dosages).
- Use **any debugging method you prefer**:
  - Print statements
  - `pdb`
  - VS Code debugger
  - Other tools
- Pass all provided **pytest** tests:
  - `test_patient_data_cleaner.py`
  - `test_med_dosage_calculator.py`
- Add comments explaining:
  - What was wrong
  - How you fixed it

### Grading

- All tests pass: **full credit**
- Clear explanations in comments
- Clean, readable code

---

## Part 2: Big Data Analysis (30%)

### Tasks

- Use `generate_large_health_data.py` to create `patients_large.csv`
- Use `analyze_large_health_data.py` with **polars** backend
- Submit:
  - Your command(s) used
  - The resulting `summary.csv`
  - A brief reflection (1 paragraph):
    - Challenges faced
    - How polars helped
    - What you learned

### Grading

- Correct output file
- Clear reflection
- Demonstrated use of polars lazy/streaming

---

## Submission Checklist

- Fixed Python scripts with comments
- Passing pytest tests
- `summary.csv` file
- Reflection paragraph

---

## Notes

<!--
Debugging skills are essential for all coding work. Handling big data is increasingly important in health data science.
-->