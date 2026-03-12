#!/usr/bin/env python3
"""Batch report for lecture 02 Polars pipeline demo.

Reads the wearable demo dataset (Parquet), builds a lazy plan, and writes a
sleep ↔ HRV report grouped by demographic columns.

Run from `lectures/02/demo/`:
    uv run python 03a_batch_report.py --config 03a_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sleep/HRV report")
    parser.add_argument("--config", default="03a_config.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    cfg = load_config(Path(args.config))

    sensor_path = cfg["inputs"]["sensor_parquet"]
    sleep_path = cfg["inputs"]["sleep_parquet"]
    users_path = cfg["inputs"]["users_parquet"]

    start_date = cfg["filters"]["start_date"]
    start_dt = datetime.fromisoformat(start_date)
    missingness_max = float(cfg["filters"]["missingness_max"])

    report_parquet = Path(cfg["outputs"]["report_parquet"])
    report_csv = Path(cfg["outputs"]["report_csv"])
    report_parquet.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning inputs lazily...")
    users = pl.scan_parquet(users_path).select(
        ["user_id", "age", "gender", "occupation"]
    )
    sleep = pl.scan_parquet(sleep_path).select(
        ["user_id", "date", "sleep_efficiency"]
    )  # keep it tight

    sensor = (
        pl.scan_parquet(sensor_path)
        .filter(pl.col("missingness_score") <= missingness_max)
        .filter(pl.col("ts_start") >= start_dt)
        .with_columns(
            pl.concat_str(
                [pl.lit("USER-"), pl.col("device_id").str.split("-").list.get(1)]
            ).alias("user_id"),
            pl.col("ts_start").dt.date().alias("date"),
            pl.col("ts_start").dt.hour().alias("hour"),
        )
    )

    # Night hours are split across midnight; keep it explicit.
    sensor_night = sensor.filter((pl.col("hour") >= 22) | (pl.col("hour") <= 6)).select(
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

    joined = sleep.join(nightly, on=["user_id", "date"], how="inner").join(
        users, on="user_id", how="left"
    )

    report = (
        joined.group_by(["occupation", "gender"])
        .agg(
            [
                pl.len().alias("n_nights"),
                pl.mean("sleep_efficiency").alias("avg_sleep_efficiency"),
                pl.mean("night_mean_sdnn").alias("avg_night_sdnn"),
                pl.corr("sleep_efficiency", "night_mean_sdnn").alias("corr_sleep_sdnn"),
            ]
        )
        .sort(["occupation", "gender"])
    )

    logger.info("Plan summary:\n%s", report.explain())

    logger.info("Collecting with streaming engine...")
    out = report.collect(engine="streaming")

    logger.info("Writing %s", report_parquet)
    out.write_parquet(report_parquet)

    logger.info("Writing %s", report_csv)
    out.write_csv(report_csv)

    logger.info("Done: %d rows", out.height)


if __name__ == "__main__":
    main()
