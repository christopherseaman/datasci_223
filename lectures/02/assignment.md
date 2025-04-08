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
  - `buggy_bmi.py` (calculates BMI and category, contains typos and logic errors)
  - `buggy_list.py` (processes a list, contains off-by-one and typo errors)
- The bugs are **typical beginner mistakes** such as:
  - Typos and wrong variable names (`NameError`)
  - Off-by-one errors (`IndexError`)
  - Incorrect logic or formulas
  - Syntax issues (`SyntaxError`, `IndentationError`)
- The code intent is **simple and easy to reason about** (e.g., calculating BMI, printing or processing list items).
- Use **any debugging method you prefer**:
  - Print statements
  - `pdb`
  - VS Code debugger
  - Other tools
- Pass all provided **pytest** tests:
  - `test_buggy_bmi.py`
  - `test_buggy_list.py`
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