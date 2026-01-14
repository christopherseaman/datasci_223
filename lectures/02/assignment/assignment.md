# Assignment 02: ICD-10 + HCPCS Prevalence (Notebook)

This notebook is the starter template for Assignment 02. Follow each section, fill in the TODOs, and run the notebook top-to-bottom.

## Goals

- Build lazy Polars scans over claims-style Parquet inputs
- Join patients, sites, and ICD-10 records
- Compute site-level prevalence of a target diagnosis
- Materialize outputs with streaming and export Parquet/CSV

## Setup

Run in a terminal:

- `uv venv .venv`
- `uv pip install -r requirements.txt`

## Generate data

- `uv run python generate_assignment_data.py --size medium --output-dir data`
- If you hit memory limits, rerun with `--size small` or `--num-patients`

## Load the config

```python
from pathlib import Path
import os
import yaml

config_path = Path(os.environ.get("POLARS_ASSIGNMENT_CONFIG", "config.yaml"))
config = yaml.safe_load(config_path.read_text())
config
```

## Part 1: Lazy scans

Create lazy scans for patients, sites, records, and lookups. Parse `record_ts` so you can filter by date later.

```python
import polars as pl

patients_lf = pl.scan_parquet(config["data"]["patients_parquet"])
# TODO: select patient_id, site_id, dob, gender, zip_code

sites_lf = pl.scan_parquet(config["data"]["sites_parquet"])
# TODO: select site_id, site_name, site_type

records_lf = pl.scan_parquet(config["data"]["records_parquet"])
# TODO: ensure record_ts is Datetime and select patient_id, site_id, record_type, code

icd_lookup_lf = pl.scan_parquet(config["data"]["icd10_lookup_parquet"])
# TODO: select code, short_description, category

hcpcs_lookup_lf = pl.scan_parquet(config["data"]["hcpcs_lookup_parquet"])
# TODO: select code, description, group (optional for context)
```

## Part 2: Filter + join

Filter to ICD-10 records that match the prefixes in the config. Use the lookup table to keep only target codes.

```python
from datetime import datetime

start_dt = datetime.fromisoformat(config["data"]["start_date"])
target_prefixes = config["data"]["target_icd_prefixes"]

icd_target_codes = (
    icd_lookup_lf
    # TODO: filter to codes with target prefixes
    # TODO: keep only the code column
)

icd_records = (
    records_lf
    # TODO: filter to ICD-10-CM records on/after start_dt
    # TODO: join to icd_target_codes on code
)

patients_with_dx = (
    icd_records
    # TODO: keep unique patient_id values
)

patients_flagged = (
    patients_lf
    # TODO: join patients_with_dx to flag has_target_dx
)

summary_lf = (
    patients_flagged
    # TODO: join sites
    # TODO: group by site_id/site_name/site_type
    # TODO: aggregate num_patients, num_patients_with_dx
    # TODO: compute pct_with_dx
    # TODO: sort by site_id
)
```

## Part 3: Materialize outputs

Collect with streaming and write the output files defined in `config.yaml`.

```python
from pathlib import Path

summary_df = summary_lf.collect(engine="streaming")

output_parquet = Path(config["outputs"]["prevalence_parquet"])
output_csv = Path(config["outputs"]["prevalence_csv"])

# TODO: create output directories
# TODO: write Parquet + CSV

summary_df
```

## Part 4: Quick checks

```python
import polars as pl

parquet_df = pl.read_parquet(output_parquet)
csv_df = pl.read_csv(output_csv)
parquet_df, csv_df
```

## Tests (run in terminal)

- `uv run pytest .github/tests -q`

## Submission Checklist

- TODOs completed and notebook runs top-to-bottom
- `outputs/site_diagnosis_prevalence.parquet` exists
- `outputs/site_diagnosis_prevalence.csv` exists
- `outputs/README.md` updated
- All tests pass locally
