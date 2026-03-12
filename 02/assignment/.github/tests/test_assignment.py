"""Autograder tests for Assignment 02: Polars EHR event logs."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import yaml


@pytest.fixture(scope="module")
def config():
    config_path = Path("config.yaml")
    assert config_path.exists(), "config.yaml not found"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_data_files_exist(config):
    data_paths = [
        config["data"]["patients_path"],
        config["data"]["sites_path"],
        config["data"]["events_path"],
        config["data"]["icd10_path"],
    ]
    missing = [path for path in data_paths if not Path(path).exists()]
    assert not missing, f"Missing data files: {missing}"


def test_outputs_exist(config):
    output_paths = [
        config["outputs"]["dx_summary_parquet"],
        config["outputs"]["dx_summary_csv"],
    ]
    missing = [path for path in output_paths if not Path(path).exists()]
    assert not missing, f"Missing output files: {missing}"


def test_dx_summary_schema(config):
    df = pl.read_parquet(config["outputs"]["dx_summary_parquet"])
    required_columns = {
        "site_id",
        "site_name",
        "site_type",
        "patients_seen",
        "diabetes_patients",
        "diabetes_prevalence",
    }
    missing = required_columns - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"


def test_dx_summary_values(config):
    df = pl.read_parquet(config["outputs"]["dx_summary_parquet"])
    assert df.height > 0, "Diagnosis summary is empty"

    assert df["patients_seen"].min() >= config["processing"]["min_patients_per_site"], (
        "patients_seen below expected minimum"
    )
    assert df["diabetes_patients"].min() >= 0, "diabetes_patients should be non-negative"
    assert df["diabetes_patients"].max() <= df["patients_seen"].max(), (
        "diabetes_patients should not exceed patients_seen"
    )

    prevalence = df["diabetes_prevalence"]
    assert prevalence.min() >= 0, "diabetes_prevalence should be >= 0"
    assert prevalence.max() <= 1, "diabetes_prevalence should be <= 1"


def test_parquet_csv_match(config):
    dx_parquet = pl.read_parquet(config["outputs"]["dx_summary_parquet"])
    dx_csv = pl.read_csv(config["outputs"]["dx_summary_csv"])
    assert dx_parquet.height == dx_csv.height, "DX parquet/csv row counts differ"
    assert set(dx_parquet.columns) == set(dx_csv.columns), (
        "Parquet and CSV have different columns"
    )


def test_prevalence_calculation_correctness(config):
    """Validate that prevalence is correctly calculated as diabetes_patients / patients_seen."""
    df = pl.read_parquet(config["outputs"]["dx_summary_parquet"])

    for row in df.to_dicts():
        expected_prevalence = row["diabetes_patients"] / row["patients_seen"]
        actual_prevalence = row["diabetes_prevalence"]
        assert abs(actual_prevalence - expected_prevalence) < 1e-6, (
            f"Site {row['site_id']}: prevalence calculation incorrect. "
            f"Expected {expected_prevalence:.6f}, got {actual_prevalence:.6f}"
        )


def test_site_join_correctness(config):
    """Verify that site names and types are correctly joined."""
    df = pl.read_parquet(config["outputs"]["dx_summary_parquet"])
    sites = pl.read_parquet(config["data"]["sites_path"])

    for site_id in df["site_id"]:
        site_row = sites.filter(pl.col("site_id") == site_id)
        assert site_row.height > 0, f"Site {site_id} found in output but not in sites table"

        output_row = df.filter(pl.col("site_id") == site_id)
        assert output_row["site_name"][0] == site_row["site_name"][0], (
            f"Site name mismatch for {site_id}"
        )
        assert output_row["site_type"][0] == site_row["site_type"][0], (
            f"Site type mismatch for {site_id}"
        )


def test_diabetes_filter_correctness(config):
    """Verify that only E11 (type 2 diabetes) codes are included."""
    events = pl.read_parquet(config["data"]["events_path"])
    prefix = config["data"]["diabetes_prefix"]
    start_date = datetime.fromisoformat(config["data"]["start_date"])

    events_filtered = events.with_columns(
        pl.col("event_ts").str.strptime(pl.Datetime, strict=False)
    ).filter(
        (pl.col("event_ts") >= start_date) &
        (pl.col("record_type") == "ICD-10-CM")
    )

    diabetes_events = events_filtered.filter(
        pl.col("code").str.starts_with(prefix)
    )

    diabetes_patient_count = diabetes_events.select("patient_id").unique().height

    output_df = pl.read_parquet(config["outputs"]["dx_summary_parquet"])
    total_diabetes_patients = output_df["diabetes_patients"].sum()

    assert total_diabetes_patients == diabetes_patient_count, (
        f"Total diabetes patients mismatch. Expected {diabetes_patient_count}, got {total_diabetes_patients}"
    )
