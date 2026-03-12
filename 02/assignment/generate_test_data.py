#!/usr/bin/env python3
"""Generate synthetic EHR data for Assignment 02 testing."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from random import Random
import re
from typing import Iterable

import polars as pl
from faker import Faker


@dataclass(frozen=True)
class CodebookConfig:
    icd10_prefixes: list[str]
    icd10_category_map: dict[str, str]
    hcpcs_groups: list[str]


def _icd10_with_dot(code_raw: str) -> str:
    code_raw = code_raw.strip()
    if len(code_raw) > 3:
        return f"{code_raw[:3]}.{code_raw[3:]}"
    return code_raw


def load_icd10_codes(path: Path, config: CodebookConfig) -> pl.DataFrame:
    """Load ICD-10 codes from the CDC order file and keep leaf codes."""
    rows: list[dict[str, str]] = []

    for line in path.read_text(encoding="latin-1").splitlines():
        if not line or line.startswith("#"):
            continue

        code_raw = line[6:13].strip()
        is_leaf = line[14:15].strip()
        short_desc = line[16:76].strip()

        if not code_raw or is_leaf != "1" or not short_desc:
            continue

        code = _icd10_with_dot(code_raw)
        if config.icd10_prefixes and not any(
            code.startswith(prefix) for prefix in config.icd10_prefixes
        ):
            continue

        category = "other"
        for prefix, label in config.icd10_category_map.items():
            if code.startswith(prefix):
                category = label
                break

        rows.append({
            "code": code,
            "short_desc": short_desc,
            "category": category,
        })

    if not rows:
        raise ValueError("No ICD-10 codes matched filters. Check prefixes.")

    return pl.DataFrame(rows).unique(subset=["code"]).sort("code")


def _is_group_header(line: str) -> bool:
    if not line.isupper():
        return False
    if line.startswith("INCLUDE") or line.startswith("EXCLUDE"):
        return False
    if line.startswith("LIST OF") or line.startswith("THIS CODE LIST"):
        return False
    return len(line) <= 80


def load_hcpcs_codes(path: Path, config: CodebookConfig) -> pl.DataFrame:
    """Load HCPCS/CPT codes and attach the most recent group header."""
    rows: list[dict[str, str]] = []
    group = "UNSPECIFIED"

    for line in path.read_text(encoding="latin-1").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        code_match = re.match(r"^[A-Z0-9]{5}\b", stripped)
        if code_match:
            code = code_match.group(0)
            desc = stripped[len(code) :].strip().strip('"')
            if not desc:
                continue
            if config.hcpcs_groups and group not in config.hcpcs_groups:
                continue
            rows.append({"code": code, "short_desc": desc, "group": group})
            continue

        if _is_group_header(stripped):
            group = stripped.strip('"')

    if not rows:
        raise ValueError("No HCPCS codes matched filters. Check group names.")

    return pl.DataFrame(rows).unique(subset=["code"]).sort("code")


def _weighted_choice(
    rng: Random, items: list[str], weights: Iterable[float]
) -> str:
    return rng.choices(items, weights=weights, k=1)[0]


def _allocate_counts(total: int, weights: list[int]) -> list[int]:
    if total <= 0:
        return [0 for _ in weights]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive value.")

    raw_counts = [total * weight / total_weight for weight in weights]
    base_counts = [int(count) for count in raw_counts]
    remainder = total - sum(base_counts)

    if remainder > 0:
        fractional = [
            (idx, raw_counts[idx] - base_counts[idx])
            for idx in range(len(weights))
        ]
        fractional.sort(key=lambda item: item[1], reverse=True)
        for idx, _ in fractional[:remainder]:
            base_counts[idx] += 1

    return base_counts


def generate_sites(
    num_patients: int, faker: Faker, rng: Random
) -> pl.DataFrame:
    """Generate site dimension table scaled by patient count."""
    if num_patients < 500:
        site_count = 3
    elif num_patients < 2000:
        site_count = 5
    else:
        site_count = 8

    site_types = [
        ("Community Clinic", 0.45),
        ("Regional Hospital", 0.35),
        ("Specialty Center", 0.2),
    ]
    site_type_names = [t[0] for t in site_types]
    site_type_weights = [t[1] for t in site_types]

    sites: list[dict[str, str | int]] = []
    for idx in range(site_count):
        site_type = _weighted_choice(rng, site_type_names, site_type_weights)
        city = faker.city()
        site_id = f"SITE-{idx + 1:03d}"

        if site_type == "Community Clinic":
            panel_size = rng.randint(200, 600)
        elif site_type == "Specialty Center":
            panel_size = rng.randint(400, 900)
        else:
            panel_size = rng.randint(700, 1800)

        sites.append({
            "site_id": site_id,
            "site_name": f"{city} {site_type}",
            "site_type": site_type,
            "zip_code": faker.postcode()[:5],
            "panel_size": int(panel_size),
        })

    return pl.DataFrame(sites)


def generate_patients(
    num_patients: int, sites_df: pl.DataFrame, faker: Faker, rng: Random
) -> pl.DataFrame:
    """Generate patient table with demographics and home site."""
    site_ids = sites_df["site_id"].to_list()
    site_weights = sites_df["panel_size"].to_list()
    site_counts = _allocate_counts(num_patients, site_weights)
    genders = ["Female", "Male", "Nonbinary"]

    patient_id_col: list[str] = []
    dob_col: list[str] = []
    gender_col: list[str] = []
    zip_col: list[str] = []
    home_site_col: list[str] = []

    patient_counter = 1
    for site_id, count in zip(site_ids, site_counts):
        if count == 0:
            continue

        patient_ids = [
            f"PAT-{patient_counter + idx:06d}" for idx in range(count)
        ]
        patient_counter += count

        dobs = [
            faker.date_of_birth(minimum_age=18, maximum_age=90).isoformat()
            for _ in range(count)
        ]
        zip_codes = [faker.postcode()[:5] for _ in range(count)]
        gender_values = rng.choices(
            genders, weights=[0.52, 0.45, 0.03], k=count
        )

        patient_id_col.extend(patient_ids)
        dob_col.extend(dobs)
        gender_col.extend(gender_values)
        zip_col.extend(zip_codes)
        home_site_col.extend([site_id] * count)

    return pl.DataFrame({
        "patient_id": patient_id_col,
        "dob": dob_col,
        "gender": gender_col,
        "zip_code": zip_col,
        "home_site_id": home_site_col,
    })


def _random_timestamp(
    rng: Random, start_dt: datetime, end_dt: datetime
) -> datetime:
    total_seconds = int((end_dt - start_dt).total_seconds())
    offset = rng.randrange(0, total_seconds)
    return start_dt + timedelta(seconds=offset)


def generate_events(
    patients_df: pl.DataFrame,
    sites_df: pl.DataFrame,
    icd10_df: pl.DataFrame,
    hcpcs_df: pl.DataFrame,
    rng: Random,
    start_date: datetime,
    end_date: datetime,
    min_events: int,
    max_events: int,
) -> pl.DataFrame:
    """Generate a long table of diagnosis and procedure events."""
    icd10_primary = icd10_df.filter(pl.col("category") == "type_2_diabetes")[
        "code"
    ].to_list()
    icd10_secondary = icd10_df.filter(pl.col("category") != "type_2_diabetes")[
        "code"
    ].to_list()
    icd10_pool = icd10_secondary or icd10_primary
    hcpcs_codes = hcpcs_df["code"].to_list()

    site_prevalence = {
        row["site_id"]: rng.uniform(0.1, 0.4)
        for row in sites_df.iter_rows(named=True)
    }

    patient_ids = patients_df["patient_id"].to_list()
    home_sites = patients_df["home_site_id"].to_list()

    event_id_col: list[str] = []
    patient_id_col: list[str] = []
    site_id_col: list[str] = []
    event_ts_col: list[str] = []
    record_type_col: list[str] = []
    code_col: list[str] = []

    record_types = ["ICD-10-CM", "HCPCS"]
    record_weights = [0.45, 0.55]
    event_counter = 1

    for patient_id, site_id in zip(patient_ids, home_sites):
        prevalence = site_prevalence.get(site_id, 0.2)
        has_diabetes = rng.random() < prevalence
        total_events = rng.randint(min_events, max_events)

        event_timestamps = [
            _random_timestamp(rng, start_date, end_date)
            for _ in range(total_events)
        ]
        record_type_values = rng.choices(
            record_types, weights=record_weights, k=total_events
        )
        if has_diabetes and total_events > 0:
            record_type_values[0] = "ICD-10-CM"

        diabetes_event_remaining = 1 if has_diabetes else 0

        for event_ts, record_type in zip(event_timestamps, record_type_values):
            if record_type == "ICD-10-CM":
                if diabetes_event_remaining > 0:
                    code = rng.choice(icd10_primary)
                    diabetes_event_remaining -= 1
                else:
                    code = rng.choice(icd10_pool)
            else:
                code = rng.choice(hcpcs_codes)

            event_id_col.append(f"EVT-{event_ts.year}-{event_counter:08d}")
            patient_id_col.append(patient_id)
            site_id_col.append(site_id)
            event_ts_col.append(event_ts.isoformat(timespec="seconds"))
            record_type_col.append(record_type)
            code_col.append(code)
            event_counter += 1

    return pl.DataFrame({
        "event_id": event_id_col,
        "patient_id": patient_id_col,
        "site_id": site_id_col,
        "event_ts": event_ts_col,
        "record_type": record_type_col,
        "code": code_col,
    })


def write_outputs(
    output_dir: Path,
    patients_df: pl.DataFrame,
    sites_df: pl.DataFrame,
    events_df: pl.DataFrame,
    icd10_df: pl.DataFrame,
    hcpcs_df: pl.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    patients_df.write_parquet(output_dir / "patients.parquet")
    sites_df.write_parquet(output_dir / "sites.parquet")
    events_df.write_parquet(output_dir / "events.parquet")
    icd10_df.write_parquet(output_dir / "icd10_codes.parquet")
    hcpcs_df.write_parquet(output_dir / "hcpcs_codes.parquet")


CODEBOOK_CONFIG = CodebookConfig(
    icd10_prefixes=["E11", "I10", "E78", "N18", "E66"],
    icd10_category_map={
        "E11": "type_2_diabetes",
        "I10": "hypertension",
        "E78": "hyperlipidemia",
        "N18": "ckd",
        "E66": "obesity",
    },
    hcpcs_groups=[
        "CLINICAL LABORATORY SERVICES",
        "RADIOLOGY AND CERTAIN OTHER IMAGING SERVICES",
        "PHYSICAL THERAPY, OCCUPATIONAL THERAPY, AND OUTPATIENT SPEECH-LANGUAGE PATHOLOGY SERVICES",
    ],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Assignment 02 test data"
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset size",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def generate_test_data(
    size: str = "small", output_dir: Path = Path("data"), seed: int = 42
) -> None:
    size_config = {
        "small": {"patients": 25000, "events": (15, 30)},
        "medium": {"patients": 100000, "events": (50, 100)},
        "large": {"patients": 500000, "events": (100, 200)},
    }

    if size not in size_config:
        raise ValueError(f"Unsupported size: {size}")

    config = size_config[size]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    faker = Faker(use_weighting=False)
    faker.seed_instance(seed)
    rng = Random(seed)

    refs_dir = Path(__file__).parent / "refs"
    icd10_path = refs_dir / "icd10cm-order-2026.txt"
    hcpcs_path = refs_dir / "2026_DHS_Code_List_Addendum_12_01_2025.txt"

    logging.info("Loading codebooks")
    icd10_df = load_icd10_codes(icd10_path, CODEBOOK_CONFIG)
    hcpcs_df = load_hcpcs_codes(hcpcs_path, CODEBOOK_CONFIG)

    logging.info("Generating sites")
    sites_df = generate_sites(config["patients"], faker, rng)

    logging.info("Generating patients")
    patients_df = generate_patients(config["patients"], sites_df, faker, rng)

    logging.info("Generating events")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    events_df = generate_events(
        patients_df,
        sites_df,
        icd10_df,
        hcpcs_df,
        rng,
        start_date,
        end_date,
        min_events=config["events"][0],
        max_events=config["events"][1],
    )

    write_outputs(
        output_dir, patients_df, sites_df, events_df, icd10_df, hcpcs_df
    )

    logging.info("Wrote %s patients", patients_df.height)
    logging.info("Wrote %s sites", sites_df.height)
    logging.info("Wrote %s events", events_df.height)
    logging.info("Output directory: %s", output_dir)


def main() -> None:
    args = parse_args()
    generate_test_data(args.size, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
