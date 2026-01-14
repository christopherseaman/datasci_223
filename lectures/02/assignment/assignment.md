# Assignment 02: Polars Lazy Pipeline (Notebook)

This notebook is the starter template for Assignment 02. Follow each section, fill in the TODOs, and run the notebook top-to-bottom.

## Goals

- Build lazy Polars scans over CSV inputs
- Join encounters + vitals and compute monthly summaries
- Materialize outputs with streaming and export Parquet/CSV
- Confirm outputs and pass the tests

## Setup

Run in a terminal:

- `uv venv .venv`
- `uv pip install -r requirements.txt`

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

Create lazy scans for encounters and vitals. Parse the date columns so you can filter and group by year/month later.

```python
import polars as pl

encounters_lf = pl.scan_csv(config["data"]["encounters_csv"])
# TODO: parse admit_ts/discharge_ts as Datetime and select key columns

vitals_lf = pl.scan_csv(config["data"]["vitals_csv"])
# TODO: parse timestamp as Datetime and cast numeric columns
```

## Part 2: Filter + join

Filter to the facilities and start date in the config. Create a patient → facility mapping to avoid duplicate joins.

```python
from datetime import datetime

start_dt = datetime.fromisoformat(config["data"]["start_date"])
facilities = config["data"]["facilities"]

encounters_filtered = encounters_lf
# TODO: filter by admit_ts >= start_dt and facilities list
# TODO: keep patient_id + facility only, then de-duplicate

vitals_filtered = vitals_lf
# TODO: filter timestamp >= start_dt

summary_lf = vitals_filtered.join(encounters_filtered, on="patient_id", how="inner")
# TODO: group by facility/year/month and aggregate num_vitals, avg_hr, avg_bmi
```

## Part 3: Materialize outputs

Collect with streaming and write the output files defined in `config.yaml`.

```python
from pathlib import Path

summary_df = summary_lf.collect(engine="streaming")

output_parquet = Path(config["outputs"]["summary_parquet"])
output_csv = Path(config["outputs"]["summary_csv"])

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
- `outputs/facility_month_summary.parquet` exists
- `outputs/facility_month_summary.csv` exists
- `outputs/README.md` updated
- All tests pass locally
