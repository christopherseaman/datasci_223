# Assignment 02: Polars on EHR Event Logs

Build a Polars pipeline that summarizes diagnosis prevalence from synthetic EHR events. Use lazy scans, filtering, joins, and group-bys to compute site-level diabetes prevalence.

## Setup

```python
import polars as pl
import yaml
from pathlib import Path
from datetime import datetime
from generate_test_data import generate_test_data

print(f"Polars version: {pl.__version__}")
print("Environment ready!")
```

## Configuration

```python
with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("Config loaded:")
print(f"  Patients: {config['data']['patients_path']}")
print(f"  Sites: {config['data']['sites_path']}")
print(f"  Events: {config['data']['events_path']}")
print(f"  ICD-10 lookup: {config['data']['icd10_path']}")
```

## Generate data

```python
SIZE = config["data"]["size"]
DATA_DIR = Path(config["data"]["dir"])

# Generate data - "medium" takes ~10 seconds on my laptop
generate_test_data(size=SIZE, output_dir=DATA_DIR)
```

## Hints (optional)

- Distinct patient counts: call `.unique()` before `group_by()`.
  - Example: `events.select(["site_id", "patient_id"]).unique()`
- Prefix filter for ICD-10: `pl.col("code").str.starts_with(prefix)`
- Optional polish: `.fill_null(0)` after a left join, and `.round(3)` on prevalence

## Part 1: Lazy Data Loading

Use `pl.scan_parquet()` to create LazyFrames without loading data into memory.

```python
# TODO: Scan patients, sites, events, and ICD-10 lookup
patients = None
sites = None
events = None
icd10 = None

# Check schemas (fast, still lazy)
if patients is not None:
    print("Patients schema:")
    print(patients.collect_schema())

if events is not None:
    print("Events schema:")
    print(events.collect_schema())
```

## Part 2: Filter and Prep Events

Filter to the assignment date window and extract ICD-10 diagnosis events.

```python
start_date = datetime.fromisoformat(config["data"]["start_date"])

# TODO: parse event_ts to Datetime
# TODO: filter events to event_ts >= start_date
# TODO: filter to record_type == "ICD-10-CM" for diagnosis events

events_filtered = None

dx_events = None
```

## Part 3: Diagnosis Prevalence by Site

Compute the percent of patients per site with a type 2 diabetes diagnosis.

```python
prefix = config["data"]["diabetes_prefix"]

# TODO: Filter dx_events to ICD-10 codes starting with prefix
# TODO: total patients per site (unique patient_id from events_filtered)
# TODO: diabetes patients per site (unique patient_id from filtered dx)
# TODO: join counts + site names, calculate prevalence

dx_summary = None

if dx_summary is not None:
    print(dx_summary.collect_schema())
```

## Part 4: Collect and Export

```python
# TODO: collect dx_summary using streaming engine
# TODO: create outputs directory
# TODO: write Parquet + CSV outputs using config paths

if dx_summary is not None:
    print("Outputs ready")
```

## Validation

```python
outputs = [
    config["outputs"]["dx_summary_parquet"],
    config["outputs"]["dx_summary_csv"],
]

missing = [path for path in outputs if not Path(path).exists()]
if missing:
    print("Missing outputs:", missing)
else:
    print("All outputs created")
```

## Next Steps (Optional)

1. Run `python -m pytest .github/tests/test_assignment.py -v` in your terminal.
2. Use exploratory data analysis (EDA) or visualization techniques to get a feel for the dataset
