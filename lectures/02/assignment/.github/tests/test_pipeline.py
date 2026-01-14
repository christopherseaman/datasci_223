from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]


def _write_test_config(tmp_path: Path) -> Path:
    config = yaml.safe_load((BASE_DIR / "config.yaml").read_text())
    config["data"]["encounters_csv"] = str(
        BASE_DIR / "data" / "encounters" / "sample.csv"
    )
    config["data"]["vitals_csv"] = str(BASE_DIR / "data" / "vitals" / "sample.csv")
    config["outputs"]["summary_parquet"] = str(
        tmp_path / "facility_month_summary.parquet"
    )
    config["outputs"]["summary_csv"] = str(tmp_path / "facility_month_summary.csv")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _run_notebook(tmp_path: Path, config_path: Path) -> Path:
    env = os.environ.copy()
    env["POLARS_ASSIGNMENT_CONFIG"] = str(config_path)

    notebook_md = BASE_DIR / "assignment.md"
    notebook_ipynb = tmp_path / "assignment.ipynb"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupytext",
            "--to",
            "notebook",
            str(notebook_md),
            "-o",
            str(notebook_ipynb),
        ],
        cwd=BASE_DIR,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--output",
            "assignment_executed.ipynb",
            str(notebook_ipynb),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    return tmp_path / "assignment_executed.ipynb"


def _expected_summary(cfg: dict) -> pl.DataFrame:
    start_dt = datetime.fromisoformat(cfg["data"]["start_date"])
    facilities = cfg["data"]["facilities"]

    encounters = (
        pl.read_csv(cfg["data"]["encounters_csv"])
        .with_columns(
            [
                pl.col("admit_ts").str.strptime(pl.Datetime, strict=False),
                pl.col("discharge_ts").str.strptime(pl.Datetime, strict=False),
            ]
        )
        .filter(pl.col("admit_ts") >= start_dt)
        .filter(pl.col("facility").is_in(facilities))
        .select(["patient_id", "facility"])
        .unique()
    )

    vitals = (
        pl.read_csv(cfg["data"]["vitals_csv"])
        .with_columns(
            [
                pl.col("timestamp").str.strptime(pl.Datetime, strict=False),
                pl.col("heart_rate").cast(pl.Float32),
                pl.col("bmi").cast(pl.Float32),
            ]
        )
        .filter(pl.col("timestamp") >= start_dt)
    )

    return (
        vitals.join(encounters, on="patient_id", how="inner")
        .group_by(
            [
                "facility",
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
            ]
        )
        .agg(
            [
                pl.len().alias("num_vitals"),
                pl.mean("heart_rate").alias("avg_hr"),
                pl.mean("bmi").alias("avg_bmi"),
            ]
        )
        .sort(["facility", "year", "month"])
    )


@pytest.fixture(scope="module")
def executed_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("assignment_run")
    config_path = _write_test_config(tmp_path)
    _run_notebook(tmp_path, config_path)
    return {
        "config_path": config_path,
        "tmp_path": tmp_path,
    }


def test_outputs_created(executed_run):
    config = yaml.safe_load(Path(executed_run["config_path"]).read_text())
    parquet_path = Path(config["outputs"]["summary_parquet"])
    csv_path = Path(config["outputs"]["summary_csv"])

    assert parquet_path.exists(), "Parquet output missing"
    assert csv_path.exists(), "CSV output missing"

    parquet_df = pl.read_parquet(parquet_path)
    csv_df = pl.read_csv(csv_path)

    assert parquet_df.height == csv_df.height
    for column in ["facility", "year", "month", "num_vitals", "avg_hr", "avg_bmi"]:
        assert column in parquet_df.columns


def test_summary_matches_expected(executed_run):
    config = yaml.safe_load(Path(executed_run["config_path"]).read_text())
    parquet_df = pl.read_parquet(config["outputs"]["summary_parquet"])

    expected = _expected_summary(config)
    result = parquet_df.select(expected.columns).sort(["facility", "year", "month"])

    assert result.to_dicts() == expected.to_dicts()
