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


def _generate_data(output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "generate_assignment_data.py"),
            "--size",
            "small",
            "--output-dir",
            str(output_dir),
            "--seed",
            "123",
        ],
        cwd=BASE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_test_config(tmp_path: Path) -> Path:
    config = yaml.safe_load((BASE_DIR / "config.yaml").read_text())

    data_dir = tmp_path / "data"
    _generate_data(data_dir)

    config["data"]["patients_parquet"] = str(data_dir / "patients.parquet")
    config["data"]["sites_parquet"] = str(data_dir / "sites.parquet")
    config["data"]["records_parquet"] = str(data_dir / "records.parquet")
    config["data"]["icd10_lookup_parquet"] = str(data_dir / "icd10_lookup.parquet")
    config["data"]["hcpcs_lookup_parquet"] = str(data_dir / "hcpcs_lookup.parquet")
    config["outputs"]["prevalence_parquet"] = str(
        tmp_path / "site_diagnosis_prevalence.parquet"
    )
    config["outputs"]["prevalence_csv"] = str(
        tmp_path / "site_diagnosis_prevalence.csv"
    )

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
    target_prefixes = cfg["data"]["target_icd_prefixes"]

    patients = pl.read_parquet(cfg["data"]["patients_parquet"])
    sites = pl.read_parquet(cfg["data"]["sites_parquet"])
    records = (
        pl.read_parquet(cfg["data"]["records_parquet"])
        .with_columns(pl.col("record_ts").cast(pl.Datetime))
        .filter(pl.col("record_ts") >= start_dt)
    )
    icd_lookup = pl.read_parquet(cfg["data"]["icd10_lookup_parquet"])

    target_codes = (
        icd_lookup.filter(pl.col("code").str.slice(0, 3).is_in(target_prefixes))
        .select("code")
        .unique()
    )

    patients_with_dx = (
        records.filter(pl.col("record_type") == "ICD-10-CM")
        .join(target_codes, on="code", how="inner")
        .select("patient_id")
        .unique()
        .with_columns(pl.lit(True).alias("has_target_dx"))
    )

    patients_flagged = patients.join(
        patients_with_dx, on="patient_id", how="left"
    ).with_columns(pl.col("has_target_dx").fill_null(False))

    return (
        patients_flagged.join(sites, on="site_id", how="left")
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
    parquet_path = Path(config["outputs"]["prevalence_parquet"])
    csv_path = Path(config["outputs"]["prevalence_csv"])

    assert parquet_path.exists(), "Parquet output missing"
    assert csv_path.exists(), "CSV output missing"

    parquet_df = pl.read_parquet(parquet_path)
    csv_df = pl.read_csv(csv_path)

    assert parquet_df.height == csv_df.height
    for column in [
        "site_id",
        "site_name",
        "site_type",
        "num_patients",
        "num_patients_with_dx",
        "pct_with_dx",
    ]:
        assert column in parquet_df.columns


def test_summary_matches_expected(executed_run):
    config = yaml.safe_load(Path(executed_run["config_path"]).read_text())
    parquet_df = pl.read_parquet(config["outputs"]["prevalence_parquet"])

    expected = _expected_summary(config)
    result = parquet_df.select(expected.columns).sort("site_id")

    expected_rows = {row["site_id"]: row for row in expected.to_dicts()}
    for row in result.to_dicts():
        expected_row = expected_rows[row["site_id"]]
        assert row["site_name"] == expected_row["site_name"]
        assert row["site_type"] == expected_row["site_type"]
        assert row["num_patients"] == expected_row["num_patients"]
        assert row["num_patients_with_dx"] == expected_row["num_patients_with_dx"]
        assert row["pct_with_dx"] == pytest.approx(
            expected_row["pct_with_dx"], abs=1e-6
        )
