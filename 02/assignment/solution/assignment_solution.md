# Assignment 02: Polars on EHR Event Logs - SOLUTION

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
patients = pl.scan_parquet(config["data"]["patients_path"])
sites = pl.scan_parquet(config["data"]["sites_path"])
events = pl.scan_parquet(config["data"]["events_path"])
icd10 = pl.scan_parquet(config["data"]["icd10_path"])

# Check schemas (fast, still lazy)
print("Patients schema:")
print(patients.collect_schema())

print("Events schema:")
print(events.collect_schema())
```

## Part 2: Filter and Prep Events

Filter to the assignment date window and extract ICD-10 diagnosis events.

```python
start_date = datetime.fromisoformat(config["data"]["start_date"])

# Parse event_ts to Datetime and filter
events_filtered = (
    events
    .with_columns(pl.col("event_ts").str.strptime(pl.Datetime, strict=False))
    .filter(pl.col("event_ts") >= start_date)
)

# Filter to ICD-10-CM diagnosis events only
dx_events = events_filtered.filter(pl.col("record_type") == "ICD-10-CM")
```

## Part 3: Diagnosis Prevalence by Site

Compute the percent of patients per site with a type 2 diabetes diagnosis.

```python
prefix = config["data"]["diabetes_prefix"]

# Filter to diabetes codes
dx_diabetes = dx_events.filter(pl.col("code").str.starts_with(prefix))

# Total patients per site (from all filtered events)
patients_by_site = (
    events_filtered
    .select(["site_id", "patient_id"])
    .unique()
    .group_by("site_id")
    .agg(pl.len().alias("patients_seen"))
)

# Diabetes patients per site
diabetes_by_site = (
    dx_diabetes
    .select(["site_id", "patient_id"])
    .unique()
    .group_by("site_id")
    .agg(pl.len().alias("diabetes_patients"))
)

# Join counts and calculate prevalence
dx_summary = (
    patients_by_site
    .join(diabetes_by_site, on="site_id", how="left")
    .with_columns(pl.col("diabetes_patients").fill_null(0))
    .with_columns(
        (pl.col("diabetes_patients") / pl.col("patients_seen")).alias("diabetes_prevalence")
    )
    .join(sites.select(["site_id", "site_name", "site_type"]), on="site_id", how="left")
    .select([
        "site_id",
        "site_name",
        "site_type",
        "patients_seen",
        "diabetes_patients",
        "diabetes_prevalence",
    ])
    .sort("site_id")
)

print(dx_summary.collect_schema())
```

## Part 4: Collect and Export

```python
# Collect with streaming
summary_df = dx_summary.collect(engine="streaming")

# Create output directory
output_dir = Path(config["outputs"]["dx_summary_parquet"]).parent
output_dir.mkdir(parents=True, exist_ok=True)

# Write outputs
summary_df.write_parquet(config["outputs"]["dx_summary_parquet"])
summary_df.write_csv(config["outputs"]["dx_summary_csv"])

print("Outputs ready")
summary_df
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

## Next Steps

Run `python -m pytest .github/tests/test_assignment.py -v` in your terminal.
