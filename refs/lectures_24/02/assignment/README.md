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
  - `patient_data_cleaner.py` (cleans and filters patient records)
  - `med_dosage_calculator.py` (calculates medication dosages)
- The first script has labeled bugs to help you get started
- The second script has more subtle bugs that require using a debugger to find
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

1. **Time Series Analysis** (15%):
   - Use the provided script to generate a large dataset of patient vitals over time
   - Implement a polars-based analysis that:
     - Identifies patients with concerning vital sign trends
     - Calculates rolling averages of vitals
     - Detects anomalies in vital signs
   - Use polars' lazy evaluation for efficiency
   - Output results as parquet files with partitioning

2. **Patient Cohort Analysis** (10%):
   - Group patients by diagnosis and demographics
   - Calculate statistics for each cohort:
     - Treatment effectiveness
     - Average length of stay
     - Readmission rates
   - Use polars' streaming capabilities for memory efficiency
   - Generate visualizations of the results

3. **Documentation** (5%):
   - Write a brief report in `analysis.md`:
     - Explain your analysis approach
     - Discuss any patterns or insights found
     - Describe how you used polars' features for efficiency
     - Include sample visualizations
     - Suggest potential clinical applications

### Grading

- Correct implementation of time series analysis
- Effective use of polars' features
- Quality of insights in cohort analysis
- Clear and insightful documentation
- Code efficiency and organization

---

## Submission Checklist

- Fixed Python scripts with bug documentation
- All pytest tests passing
- Time series analysis implementation
- Cohort analysis results
- `analysis.md` with documentation
- Generated parquet files and visualizations

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
The debugging portion teaches systematic debugging with increasing difficulty.
The big data portion focuses on real-world health data analysis scenarios.
-->