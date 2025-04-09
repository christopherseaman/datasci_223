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
  - What was wrong (use the comment format: `# BUG: description of the bug`)
  - How you fixed it (use the comment format: `# FIX: description of the fix`)
- **Important for autograding**: Do not change function names or return types

### Grading

- All tests pass: **full credit**
- Clear explanations in comments
- Clean, readable code

---
## Part 2: Big Data Analysis (30%)

### Tasks

1. **Data Conversion and Comparison** (10%):
   - Use `generate_large_health_data.py` to create `patients_large.csv`
   - Convert the CSV dataset to Parquet format
   - Measure and report the file size difference
   - Create a `format_comparison.csv` with columns for format, size, and load time

2. **Data Analysis** (15%):
   - Use `analyze_large_health_data.py` with **polars** backend
   - Implement lazy evaluation and streaming techniques
   - Produce a `results.parquet` file with your analysis results

3. **Documentation** (5%):
   - Write a brief reflection (1 paragraph) in `reflection.md`:
     - Challenges faced with large datasets
     - How polars and Parquet improved your workflow
     - Performance differences observed
     - What you learned
    - What you learned

### Grading

- Correct output files with exact required names
- Clear reflection with specific insights
- Demonstrated use of polars lazy/streaming techniques
- Proper implementation of Parquet conversion and comparison

---

## Submission Checklist

- Fixed Python scripts with standardized bug/fix comments
- All pytest tests passing
- `format_comparison.csv` file with correct columns
- `results.parquet` file with analysis results
- `reflection.md` file with your insights

### Autograding Requirements

For successful autograding:
1. Do not rename any functions or files
2. Follow the exact file naming conventions specified
3. Ensure your code runs without errors using the provided test scripts
4. Use the required comment formats for bug documentation
5. Make sure all output files have the exact column names specified

---

## Notes

<!--
Debugging skills are essential for all coding work. Handling big data is increasingly important in health data science.
-->