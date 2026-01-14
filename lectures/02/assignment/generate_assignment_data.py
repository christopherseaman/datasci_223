#!/usr/bin/env python3
"""
Assignment data generator for ICD-10 + HCPCS workflows.

Generates patient demographics, site rosters, long-format diagnosis/procedure
records, and code lookup tables. The generator draws real ICD-10-CM and
HCPCS/CPT-style codes from the reference files in refs/.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import yaml
from faker import Faker

BASE_DIR = Path(__file__).parent
REFS_DIR = BASE_DIR / "refs"
DATA_DICT_PATH = BASE_DIR / "data_dictionary.yaml"
ICD10_REF_PATH = REFS_DIR / "icd10cm-order-2026.txt"
HCPCS_REF_PATH = REFS_DIR / "2026_DHS_Code_List_Addendum_12_01_2025.txt"

DATA_DICT = yaml.safe_load(DATA_DICT_PATH.read_text())
POPULATION = DATA_DICT["population"]
SITE_RANGES = DATA_DICT["site_size_ranges"]
DEFAULTS = DATA_DICT["generation_defaults"]
PATIENTS_BY_SIZE = DEFAULTS["patients_by_size"]
DEFAULT_SIZE = DEFAULTS["default_size"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ICD/HCPCS assignment data")
    parser.add_argument(
        "--size",
        choices=sorted(PATIENTS_BY_SIZE.keys()),
        default=DEFAULT_SIZE,
        help="Preset patient counts from data_dictionary.yaml",
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=None,
        help="Override patient count (bypasses --size)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for generated Parquet files",
    )
    return parser.parse_args()


def format_icd10_code(raw_code: str) -> str:
    if len(raw_code) <= 3:
        return raw_code
    return f"{raw_code[:3]}.{raw_code[3:]}"


def load_icd10_codes(prefixes: Iterable[str], max_codes: int) -> list[dict]:
    prefixes = tuple(prefixes)
    rows = []

    for line in ICD10_REF_PATH.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        raw_code = line[6:13].strip()
        if not raw_code:
            continue
        if line[14:15] != "1":
            continue
        if not raw_code.startswith(prefixes):
            continue

        short_desc = line[16:76].strip()
        long_desc = line[77:].strip() if len(line) > 77 else short_desc
        code = format_icd10_code(raw_code)

        rows.append(
            {
                "code": code,
                "short_description": short_desc,
                "long_description": long_desc,
                "category": raw_code[:3],
            }
        )

        if len(rows) >= max_codes:
            break

    return rows


def parse_hcpcs_codes() -> list[dict]:
    codes = []
    current_group = "Uncategorized"

    group_pattern = re.compile(r"^[A-Z][A-Z /&-]+$")
    code_pattern = re.compile(r"^(?P<code>[0-9A-Z]{5})\s+(?P<desc>.+)$")

    for raw_line in HCPCS_REF_PATH.read_text(encoding="latin-1").splitlines():
        line = raw_line.strip().strip('"')
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith(("LIST OF", "THIS CODE LIST")):
            continue
        if line.upper().startswith(("INCLUDE", "EXCLUDE")):
            continue

        if group_pattern.match(line) and "CODE" not in line:
            current_group = line
            continue

        match = code_pattern.match(line)
        if match:
            codes.append(
                {
                    "code": match.group("code"),
                    "description": match.group("desc").strip(),
                    "group": current_group,
                }
            )

    return codes


def filter_hcpcs_codes(
    codes: list[dict], keywords: Iterable[str], max_codes: int
) -> list[dict]:
    keywords = [keyword.lower() for keyword in keywords]
    seen = set()

    def matches_keywords(row: dict) -> bool:
        description = row["description"].lower()
        return any(keyword in description for keyword in keywords)

    filtered = [row for row in codes if matches_keywords(row)]
    if len(filtered) < max_codes:
        filtered.extend(
            [row for row in codes if row not in filtered][: max_codes - len(filtered)]
        )

    selected = []
    for row in filtered:
        if row["code"] in seen:
            continue
        selected.append(row)
        seen.add(row["code"])
        if len(selected) >= max_codes:
            break

    return selected


def generate_sites(
    num_patients: int, faker: Faker, rng: np.random.Generator
) -> pl.DataFrame:
    site_count = max(3, min(10, num_patients // 1500 + 1))
    site_rows = []

    for index in range(site_count):
        site_type = rng.choice(list(SITE_RANGES.keys())).item()
        size_min = SITE_RANGES[site_type]["min"]
        size_max = SITE_RANGES[site_type]["max"]
        capacity = int(rng.integers(size_min, size_max + 1))

        site_rows.append(
            {
                "site_id": f"SITE-{index + 1:03d}",
                "site_name": f"{faker.city()} {site_type}",
                "site_type": site_type,
                "patient_capacity": capacity,
            }
        )

    return pl.DataFrame(site_rows)


def generate_patients(
    num_patients: int,
    sites_df: pl.DataFrame,
    faker: Faker,
    rng: np.random.Generator,
) -> pl.DataFrame:
    site_ids = sites_df.get_column("site_id").to_list()
    weights = sites_df.get_column("patient_capacity").to_list()
    probabilities = np.array(weights) / sum(weights)

    assignments = rng.choice(site_ids, size=num_patients, p=probabilities)

    patients = []
    for index in range(num_patients):
        patients.append(
            {
                "patient_id": f"PAT-{index + 1:06d}",
                "site_id": assignments[index],
                "dob": faker.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
                "gender": faker.random_element(["F", "M", "X"]),
                "zip_code": faker.postcode(),
            }
        )

    return pl.DataFrame(patients)


def generate_records(
    patients_df: pl.DataFrame,
    icd_target_codes: list[str],
    icd_related_codes: list[str],
    hcpcs_codes: list[str],
    rng: np.random.Generator,
) -> pl.DataFrame:
    records = []
    record_counter = 1
    base_date = datetime.now() - timedelta(days=540)

    for patient in patients_df.iter_rows(named=True):
        patient_id = patient["patient_id"]
        site_id = patient["site_id"]
        has_target = rng.random() < POPULATION["target_prevalence"]

        dx_count = int(rng.integers(1, 4))
        proc_count = int(rng.integers(1, 3))

        dx_codes = []
        if has_target:
            dx_codes.append(rng.choice(icd_target_codes).item())
            if dx_count > 1:
                dx_codes.extend(
                    rng.choice(
                        icd_related_codes, size=dx_count - 1, replace=True
                    ).tolist()
                )
        else:
            dx_codes.extend(
                rng.choice(icd_related_codes, size=dx_count, replace=True).tolist()
            )

        for code in dx_codes:
            record_ts = base_date + timedelta(
                days=int(rng.integers(0, 540)),
                hours=int(rng.integers(0, 24)),
            )
            records.append(
                {
                    "record_id": f"REC-{record_ts.year}-{record_counter:08d}",
                    "patient_id": patient_id,
                    "site_id": site_id,
                    "record_ts": record_ts,
                    "record_type": "ICD-10-CM",
                    "code": code,
                }
            )
            record_counter += 1

        for _ in range(proc_count):
            record_ts = base_date + timedelta(
                days=int(rng.integers(0, 540)),
                hours=int(rng.integers(0, 24)),
            )
            records.append(
                {
                    "record_id": f"REC-{record_ts.year}-{record_counter:08d}",
                    "patient_id": patient_id,
                    "site_id": site_id,
                    "record_ts": record_ts,
                    "record_type": "HCPCS",
                    "code": rng.choice(hcpcs_codes).item(),
                }
            )
            record_counter += 1

    return pl.DataFrame(records)


def main() -> None:
    args = parse_args()
    size_label = args.size
    num_patients = args.num_patients or PATIENTS_BY_SIZE[size_label]

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    faker = Faker("en_US")
    faker.seed_instance(args.seed)
    rng = np.random.default_rng(args.seed)

    logger.info("Loading ICD-10-CM + HCPCS references...")
    icd_target = load_icd10_codes(
        POPULATION["target_icd_prefixes"], POPULATION["icd_target_max"]
    )
    icd_related = load_icd10_codes(
        POPULATION["related_icd_prefixes"], POPULATION["icd_related_max"]
    )

    hcpcs_raw = parse_hcpcs_codes()
    hcpcs_selected = filter_hcpcs_codes(
        hcpcs_raw, POPULATION["hcpcs_keywords"], POPULATION["hcpcs_max"]
    )

    icd_target_df = pl.DataFrame(icd_target)
    icd_related_df = pl.DataFrame(icd_related)
    if icd_related_df.is_empty():
        icd_related_df = icd_target_df

    icd_lookup_df = pl.concat([icd_target_df, icd_related_df]).unique(subset=["code"])
    hcpcs_lookup_df = pl.DataFrame(hcpcs_selected).unique(subset=["code"])

    if icd_lookup_df.is_empty() or hcpcs_lookup_df.is_empty():
        raise SystemExit("Reference code parsing failed; check refs/ inputs.")

    logger.info(
        "Generating sites + patients (%s size, %s patients)...",
        size_label,
        num_patients,
    )
    sites_df = generate_sites(num_patients, faker, rng)
    patients_df = generate_patients(num_patients, sites_df, faker, rng)

    logger.info("Generating diagnosis + procedure records...")
    records_df = generate_records(
        patients_df,
        icd_target_codes=icd_target_df.get_column("code").to_list(),
        icd_related_codes=icd_related_df.get_column("code").to_list(),
        hcpcs_codes=hcpcs_lookup_df.get_column("code").to_list(),
        rng=rng,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing Parquet outputs...")
    sites_df.write_parquet(output_dir / "sites.parquet", compression="snappy")
    patients_df.write_parquet(output_dir / "patients.parquet", compression="snappy")
    records_df.write_parquet(
        output_dir / "records.parquet", compression="snappy", row_group_size=50_000
    )
    icd_lookup_df.write_parquet(
        output_dir / "icd10_lookup.parquet", compression="snappy"
    )
    hcpcs_lookup_df.write_parquet(
        output_dir / "hcpcs_lookup.parquet", compression="snappy"
    )

    metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "size": size_label,
        "num_patients": num_patients,
        "num_sites": sites_df.height,
        "num_records": records_df.height,
        "icd_codes": icd_lookup_df.height,
        "hcpcs_codes": hcpcs_lookup_df.height,
    }

    metadata_path = output_dir / "generation_metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata))

    logger.info("Generation complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
