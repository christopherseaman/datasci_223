# Assignment 02: ICD-10 + HCPCS Prevalence (Notebook)

**Due:** Before Lecture 03
**Points:** Pass/Fail (autograded)
**Skills:** Polars lazy scans, joins, filtering with code lookups, streaming collect

## Overview

Use a notebook to build a lazy Polars pipeline on claims-style Parquet files. You will:

1. Load paths + target ICD-10 prefixes from `config.yaml`
2. Scan patients, sites, records, and lookup tables lazily
3. Filter diagnosis records to the target ICD-10 prefixes
4. Compute site-level prevalence of the target diagnosis
5. Collect with streaming and write Parquet + CSV artifacts

All tests live under `.github/tests` inside this folder. Push frequently to see GitHub Classroom feedback.

## Assignment Structure

```
assignment/
├── assignment.md                 # Notebook-friendly instructions (source)
├── assignment.ipynb              # Generated from assignment.md
├── config.yaml                   # Centralized inputs/outputs
├── requirements.txt              # Packages for notebooks + tests
├── data/
│   ├── patients.parquet          # Patient demographics
│   ├── sites.parquet             # Site roster
│   ├── records.parquet           # Long-format ICD/HCPCS records
│   ├── icd10_lookup.parquet      # ICD-10 code lookup
│   └── hcpcs_lookup.parquet      # HCPCS/CPT code lookup
├── refs/
│   ├── icd10cm-order-2026.txt    # ICD-10-CM reference file
│   └── 2026_DHS_Code_List_Addendum_12_01_2025.txt
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

- `uv venv .venv`
- `uv pip install -r requirements.txt`

Open `assignment.ipynb` in VS Code (generated from `assignment.md`) and complete the TODOs.

## Data

Generate the assignment data from the reference code lists:

- `uv run python generate_assignment_data.py --size medium --output-dir data`

If you hit memory limits, rerun with `--size small` or pass `--num-patients` directly.
## Tasks

You will implement the notebook sections to:

- Build lazy scans with `pl.scan_parquet` and parse `record_ts`
- Filter ICD-10 records using the prefixes in `config.yaml`
- Join patients + sites to compute site-level prevalence
- Write `outputs/site_diagnosis_prevalence.parquet` + `.csv`

## Tests

Run locally from this assignment folder:

- `uv run pytest .github/tests -q`

Passing locally mirrors GitHub Classroom.

## Submission Checklist

- [ ] Notebook TODOs completed and run top-to-bottom
- [ ] `outputs/site_diagnosis_prevalence.parquet` exists
- [ ] `outputs/site_diagnosis_prevalence.csv` exists
- [ ] `outputs/README.md` updated
- [ ] All tests in `.github/tests` pass locally
- [ ] Push to GitHub Classroom and confirm CI success
