# 03a_batch_report: Reproducible batch pipeline (CLI)

Goal: run a small, reproducible Polars batch job that reads the generated wearable tables, joins them, and writes a report artifact.

In notebook form, we’ll implement the same pipeline in cells (using the same config), then display the output table and an inline plot.

This is the same mental model as a real lab pipeline: config-driven inputs, lazy scans, and streaming output.

This demo has two representations:

- The implementation is the script: `03a_batch_report.py`.
- This notebook is a walkthrough with additional commentary.

## Files

- `03a_batch_report.py` (script)
- `03a_config.yaml` (config)

## Data setup

We’ll generate the dataset if it’s missing.

```python
from pathlib import Path
import subprocess
import sys

Path("data").mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(parents=True, exist_ok=True)

sensor_dir = Path("data/sensor_hrv")
if not sensor_dir.exists() or not list(sensor_dir.glob("*.parquet")):
    subprocess.run(
        [sys.executable, "generate_demo_data.py", "--size", "small", "--output-dir", "data"],
        check=True,
    )
```

## 1) Configure inputs/filters/outputs (YAML)

We’ll read `03a_config.yaml` so the paths and filters are not hardcoded.

```python
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("03a_config.yaml").read_text())
cfg
```

## 2) Load phase (lazy scans)

The important point is that all three tables are scanned lazily.

```python
import polars as pl

sensor = pl.scan_parquet(cfg["inputs"]["sensor_parquet"])
sleep = pl.scan_parquet(cfg["inputs"]["sleep_parquet"])
users = pl.scan_parquet(cfg["inputs"]["users_parquet"])

sensor.collect_schema()
```

## 3) Transform phase (keys → aggregate → joins)

We’ll:

1. filter the big sensor table early
2. derive `user_id` + `date` so it can join to `sleep_diary`
3. aggregate to one row per user-night (this keeps the join small)

```python
from datetime import datetime

start_dt = datetime.fromisoformat(cfg["filters"]["start_date"])
missingness_max = float(cfg["filters"]["missingness_max"])

sensor_keyed = (
    sensor
    .filter(pl.col("missingness_score") <= missingness_max)
    .filter(pl.col("ts_start") >= start_dt)
    .with_columns(
        pl.concat_str([pl.lit("USER-"), pl.col("device_id").str.split("-").list.get(1)]).alias("user_id"),
        pl.col("ts_start").dt.date().alias("date"),
        pl.col("ts_start").dt.hour().alias("hour"),
    )
)

sensor_night = sensor_keyed.filter((pl.col("hour") >= 22) | (pl.col("hour") <= 6)).select(
    ["user_id", "date", "heart_rate", "hrv_sdnn", "hrv_rmssd", "steps"]
)

nightly = sensor_night.group_by(["user_id", "date"]).agg(
    [
        pl.len().alias("num_segments"),
        pl.mean("heart_rate").alias("night_mean_hr"),
        pl.mean("hrv_sdnn").alias("night_mean_sdnn"),
        pl.mean("hrv_rmssd").alias("night_mean_rmssd"),
        pl.sum("steps").alias("night_steps"),
    ]
)

joined = (
    sleep.select(["user_id", "date", "sleep_efficiency"])
    .join(nightly, on=["user_id", "date"], how="inner")
    .join(users.select(["user_id", "age", "gender", "occupation"]), on="user_id", how="left")
)

report = (
    joined.group_by(["occupation", "gender"]).agg(
        [
            pl.len().alias("n_nights"),
            pl.mean("sleep_efficiency").alias("avg_sleep_efficiency"),
            pl.mean("night_mean_sdnn").alias("avg_night_sdnn"),
            pl.corr("sleep_efficiency", "night_mean_sdnn").alias("corr_sleep_sdnn"),
        ]
    )
    .sort(["occupation", "gender"])
)

print(report.explain())
```

## 4) Materialize phase (streaming collect + write artifacts)

```python
out = report.collect(engine="streaming")

parquet_path = Path(cfg["outputs"]["report_parquet"])
csv_path = Path(cfg["outputs"]["report_csv"])

parquet_path.parent.mkdir(parents=True, exist_ok=True)
out.write_parquet(parquet_path)
out.write_csv(csv_path)

out.head(10)
```

## 5) Quick validation

```python
assert out.height > 0
assert {"occupation", "gender", "n_nights"}.issubset(set(out.columns))
assert out["n_nights"].min() > 0
assert out["avg_sleep_efficiency"].is_finite().all()
assert out["avg_night_sdnn"].is_finite().all()
"OK: output looks sane"
```


Watch the logs:

- “Scanning inputs lazily…” (load phase)
- “Plan summary…” (transform phase)
- “Collecting with streaming engine…” (materialize phase)
- “Writing …parquet / …csv” (artifact emission)

## 3) Confirm artifacts exist

```bash
ls -lh outputs/
```

Expected:

- `outputs/sleep_hrv_report.parquet`
- `outputs/sleep_hrv_report.csv`

## 4) Validate the output (fast sanity checks)

Run this immediately after the pipeline finishes:

```bash
uv run python - <<'PY'
import polars as pl

df = pl.read_parquet("outputs/sleep_hrv_report.parquet")
print(df.head())

assert df.height > 0
assert {"occupation", "gender", "n_nights"}.issubset(set(df.columns))
assert df["n_nights"].min() > 0

# Basic plausibility checks (not strict scientific claims)
assert df["avg_sleep_efficiency"].is_finite().all()
assert df["avg_night_sdnn"].is_finite().all()

print("OK: output looks sane")
PY
```

## Visual: group summary

```python
import polars as pl
import altair as alt

report = pl.read_parquet("outputs/sleep_hrv_report.parquet")
report.head()
```

```python
chart_df = report.to_pandas()

alt.Chart(chart_df).mark_bar().encode(
    x=alt.X("occupation:N", sort=None),
    y=alt.Y("avg_night_sdnn:Q", title="Avg night SDNN"),
    color="gender:N",
    tooltip=["n_nights", "avg_sleep_efficiency", "avg_night_sdnn"],
).properties(width=700, height=300)
```

## Checkpoints

- The script logs the query plan and output paths.
- The parquet output has one row per `(occupation, gender)` group.
- The report includes both sleep and physiology columns.
