# Debugging Assignment Plan

---

## Overview

Create two debugging assignments with:

- Clean, well-commented versions
- Buggy versions with typical beginner errors
- Sample data files (JSON)
- Pytest tests for each

---

## Directory Structure

```
lectures/02/assignment/
├── data/
│   ├── patients.json
│   └── meds.json
├── patient_data_cleaner.py            # clean version, well-commented
├── patient_data_cleaner_buggy.py      # buggy version, same comments
├── test_patient_data_cleaner.py
├── med_dosage_calculator.py           # clean version, well-commented
├── med_dosage_calculator_buggy.py     # buggy version, same comments
├── test_med_dosage_calculator.py
```

---

## Script 1: Patient Data Cleaner

### Purpose

- Load patient records from `patients.json`
- Capitalize names
- Convert ages to integers
- Filter out patients under 18
- Print cleaned list

### Bugs to Insert

- Typos in keys
- Forget to convert age to int
- Off-by-one in filtering
- Logic error: keep under-18 instead of removing
- Syntax error (missing colon)

---

## Script 2: Medication Dosage Calculator

### Purpose

- Load patient records from `meds.json`
- Lookup dosage factor by medication
- Calculate dosage = weight * factor
- Print patient name and dosage
- Sum total medication needed

### Bugs to Insert

- Typos in medication names
- Wrong factor used
- Arithmetic error
- Skip some patients
- Sum total incorrectly
- Syntax error

---

## Tests

- Verify cleaned patient list only includes 18+
- Names are capitalized
- Ages are integers
- Dosages are correct
- Total medication sum is correct

---

## Additional Notes

- Sample data stored in JSON files
- Buggy versions have same comments as clean versions
- Students can use any debugging method
- Focus is on typical beginner bugs, not algorithm design