# Assignment 02: Polars Lazy Pipeline (Notebook)

**Due:** Before Lecture 03
**Points:** Pass/Fail (autograded)
**Skills:** Polars lazy scans, joins, streaming collect, Parquet/CSV outputs

## Overview

Use a notebook to build a lazy Polars pipeline on the provided CSVs. You will:

1. Load paths and filters from `config.yaml`
2. Scan encounters + vitals lazily and parse timestamps
3. Join on `patient_id` and aggregate monthly facility summaries
4. Collect with streaming and write Parquet + CSV artifacts

All tests live under `.github/tests` inside this folder. Push frequently to see GitHub Classroom feedback.

## Assignment Structure

```
assignment/
├── assignment.md                 # Notebook-friendly instructions (source)
├── assignment.ipynb              # Generated from assignment.md
├── config.yaml                   # Centralized inputs/outputs
├── requirements.txt              # Packages for notebooks + tests
├── data/
│   ├── encounters/sample.csv     # Small encounter log
│   └── vitals/sample.csv         # Small vitals file
├── outputs/
│   ├── README.md                 # Describe generated artifacts
│   └── (created by you)
├── hints.md                      # Optional hints
└── .github/
    ├── tests/test_pipeline.py    # Autograder tests
    └── workflows/classroom.yml   # Do not modify
```

## Setup

From this assignment folder:

```bash
uv venv .venv
uv pip install -r requirements.txt
```

Open `assignment.ipynb` in VS Code (generated from `assignment.md`) and complete the TODOs.

## Data

Sample CSVs are already provided under `data/`. No download required.

## Tasks

You will implement the notebook sections to:

- Build lazy scans with `pl.scan_csv` and parse timestamps
- Filter by `start_date` and `facilities` from the config
- Create a patient → facility mapping to avoid duplicate joins
- Aggregate to `facility`, `year`, `month` with `num_vitals`, `avg_hr`, `avg_bmi`
- Collect with streaming and write `outputs/facility_month_summary.parquet` + `.csv`

## Tests

Run locally from this assignment folder:

```bash
uv run pytest .github/tests -q
```

Passing locally mirrors GitHub Classroom.

## Submission Checklist

- [ ] Notebook TODOs completed and run top-to-bottom
- [ ] `outputs/facility_month_summary.parquet` exists
- [ ] `outputs/facility_month_summary.csv` exists
- [ ] `outputs/README.md` updated
- [ ] All tests in `.github/tests` pass locally
- [ ] Push to GitHub Classroom and confirm CI success
