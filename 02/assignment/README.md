# Assignment 02: Polars on EHR Event Logs

**Due:** Before Lecture 03  
**Points:** Pass/Fail (autograded)  
**Skills:** Polars lazy execution, joins, group-by aggregation, streaming collection

## Overview

You will analyze synthetic EHR events for a type 2 diabetes-focused population. The data includes:

- **Patients** (demographics + home site)
- **Sites** (clinics/hospitals)
- **Events** (ICD-10 diagnoses)
- **Code lookups** (ICD-10 dictionary)

Your task is to build a lazy Polars pipeline that summarizes **diabetes diagnosis prevalence** by site.

## Assignment Structure

```
assignment/
├── assignment.ipynb           # YOUR WORK GOES HERE
├── assignment.md              # Notebook source (jupytext)
├── assignment_solution.md     # Instructor solution
├── config.yaml                # Paths + filters
├── data/
│   └── schema.yaml            # Data schema reference
├── .github/
│   └── tests/test_assignment.py
├── requirements.txt
└── generate_test_data.py
```

## Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python3 generate_test_data.py --size small --output-dir data
```

This creates:
- `data/patients.parquet`
- `data/sites.parquet`
- `data/events.parquet`
- `data/icd10_codes.parquet`

The generator reads source dictionaries from `refs/raw/` (included in the repo).

### 3. Open the notebook

```bash
jupyter notebook assignment.ipynb
```

## Tasks

Complete all cells marked `# TODO` in `assignment.ipynb`:

1. **Lazy scans** for patients, sites, events, and ICD-10 lookup
2. **Filtering** to the date window and ICD-10 diabetes codes
3. **Joining + aggregation** to compute site-level prevalence
4. **Streaming export** to Parquet + CSV outputs

## Testing

Run the autograder locally:

```bash
pytest .github/tests/test_assignment.py -v
```

If all tests pass, commit and push to GitHub Classroom.
