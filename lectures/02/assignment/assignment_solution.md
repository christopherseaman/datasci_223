# Assignment 02: ICD-10 + HCPCS Prevalence (Solution)

This notebook shows a completed reference solution for Assignment 02.

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

```python
import polars as pl

patients_lf = pl.scan_parquet(config["data"]["patients_parquet"]).select(
    ["patient_id", "site_id", "dob", "gender", "zip_code"]
)

sites_lf = pl.scan_parquet(config["data"]["sites_parquet"]).select(
    ["site_id", "site_name", "site_type"]
)

records_lf = (
    pl.scan_parquet(config["data"]["records_parquet"])
    .with_columns(pl.col("record_ts").cast(pl.Datetime))
    .select(["patient_id", "site_id", "record_ts", "record_type", "code"])
)

icd_lookup_lf = pl.scan_parquet(config["data"]["icd10_lookup_parquet"]).select(
    ["code", "short_description", "category"]
)

hcpcs_lookup_lf = pl.scan_parquet(config["data"]["hcpcs_lookup_parquet"]).select(
    ["code", "description", "group"]
)

hcpcs_lookup_lf.collect().head()
```

## Part 2: Filter + join

```python
from datetime import datetime

start_dt = datetime.fromisoformat(config["data"]["start_date"])
target_prefixes = config["data"]["target_icd_prefixes"]

icd_target_codes = (
    icd_lookup_lf
    .filter(pl.col("code").str.slice(0, 3).is_in(target_prefixes))
    .select("code")
    .unique()
)

icd_records = (
    records_lf
    .filter(pl.col("record_type") == "ICD-10-CM")
    .filter(pl.col("record_ts") >= start_dt)
    .join(icd_target_codes, on="code", how="inner")
)

patients_with_dx = (
    icd_records
    .select("patient_id")
    .unique()
    .with_columns(pl.lit(True).alias("has_target_dx"))
)

patients_flagged = (
    patients_lf
    .join(patients_with_dx, on="patient_id", how="left")
    .with_columns(pl.col("has_target_dx").fill_null(False))
)

summary_lf = (
    patients_flagged
    .join(sites_lf, on="site_id", how="left")
    .group_by(["site_id", "site_name", "site_type"])
    .agg(
        [
            pl.len().alias("num_patients"),
            pl.sum("has_target_dx").alias("num_patients_with_dx"),
        ]
    )
    .with_columns(
        (pl.col("num_patients_with_dx") / pl.col("num_patients")).alias(
            "pct_with_dx"
        )
    )
    .sort("site_id")
)

summary_lf.explain()
```

## Part 3: Materialize outputs

```python
from pathlib import Path

summary_df = summary_lf.collect(engine="streaming")

output_parquet = Path(config["outputs"]["prevalence_parquet"])
output_csv = Path(config["outputs"]["prevalence_csv"])

output_parquet.parent.mkdir(parents=True, exist_ok=True)
output_csv.parent.mkdir(parents=True, exist_ok=True)

summary_df.write_parquet(output_parquet)
summary_df.write_csv(output_csv)

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
