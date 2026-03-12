#!/usr/bin/env python3
"""
Generate synthetic patient data for Lecture 01 demos.

Demonstrates CONFIG-DRIVEN DEVELOPMENT:
- All parameters in data_generation_config.yaml (not hardcoded)
- Uses Faker library for realistic names
- Reproducible with random seed
- Easy to adjust distributions without changing code

Usage:
    python generate_demo_data.py

Outputs:
    - data/patient_intake.csv (clean, realistic data)
    - data/patient_intake_missing_height.csv (schema violation)
    - data/patient_intake_bad_values.csv (bounds violations)
"""

import random
from pathlib import Path

import pandas as pd
import yaml
from faker import Faker


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_patient_id(idx: int, prefix: str = "P") -> str:
    """Generate patient ID like P001, P002, etc."""
    return f"{prefix}{idx:03d}"


def generate_clean_patients(n: int, config: dict, fake: Faker) -> pd.DataFrame:
    """
    Generate realistic patient intake data using config parameters.

    Uses Faker for names and config-driven distributions for demographics.
    """
    patients = []
    demo_config = config["demographics"]
    bounds = config["bounds"]
    age_range = demo_config["age_range"]

    for i in range(1, n + 1):
        # Randomly assign sex
        sex = random.choice(["M", "F"])

        # Get sex-specific parameters from config
        sex_key = "male" if sex == "M" else "female"
        weight_params = demo_config[sex_key]["weight_kg"]
        height_params = demo_config[sex_key]["height_cm"]

        # Generate name using Faker (sex-appropriate)
        if sex == "M":
            first_name = fake.first_name_male()
        else:
            first_name = fake.first_name_female()
        last_name = fake.last_name()

        # Generate anthropometric values from normal distributions, clip to bounds
        weight = random.gauss(weight_params["mean"], weight_params["std"])
        weight = max(bounds["weight_kg"]["min"],
                    min(bounds["weight_kg"]["max"], weight))

        height = random.gauss(height_params["mean"], height_params["std"])
        height = max(bounds["height_cm"]["min"],
                    min(bounds["height_cm"]["max"], height))

        age = random.randint(age_range["min"], age_range["max"])

        patients.append({
            "patient_id": generate_patient_id(i),
            "first_name": first_name,
            "last_name": last_name,
            "weight_kg": round(weight, 1),
            "height_cm": int(height),
            "age": age,
            "sex": sex,
        })

    return pd.DataFrame(patients)


def create_missing_column_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remove specified column from config to test schema validation."""
    column_to_remove = config["column_to_remove"]
    n_samples = config["sample_sizes"]["missing_column"]

    # Create copy without the specified column
    remaining_cols = [col for col in df.columns if col != column_to_remove]
    return df[remaining_cols].head(n_samples).copy()


def create_bad_values_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Introduce violations specified in config to test bounds checking."""
    n_samples = config["sample_sizes"]["bad_values"]
    bad_df = df.head(n_samples).copy()

    # Apply violations from config
    for violation in config["violations"]:
        idx = violation["patient_index"]
        if idx < len(bad_df):
            bad_df.loc[idx, violation["column"]] = violation["value"]
            print(f"  → Row {idx}: {violation['column']}={violation['value']} "
                  f"({violation['reason']})")

    return bad_df


def main():
    """Generate all demo data files using config parameters."""
    # Load configuration (config-driven!)
    config_path = Path(__file__).parent / "data_generation_config.yaml"
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path.name}")

    # Set random seed for reproducibility
    random.seed(config["random_seed"])
    fake = Faker()
    fake.seed_instance(config["random_seed"])

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating synthetic patient data (seed={config['random_seed']})...")

    # Generate clean data
    n_clean = config["sample_sizes"]["clean_data"]
    clean_df = generate_clean_patients(n_clean, config, fake)
    clean_path = output_dir / config["output_files"]["clean"]
    clean_df.to_csv(clean_path, index=False)
    print(f"✓ Created {clean_path.name}: {len(clean_df)} patients")

    # Generate missing column data
    missing_df = create_missing_column_data(clean_df, config)
    missing_path = output_dir / config["output_files"]["missing_column"]
    missing_df.to_csv(missing_path, index=False)
    print(f"✓ Created {missing_path.name}: missing '{config['column_to_remove']}' column")

    # Generate bad values data
    print(f"✓ Creating {config['output_files']['bad_values']}: introducing violations")
    bad_df = create_bad_values_data(clean_df, config)
    bad_path = output_dir / config["output_files"]["bad_values"]
    bad_df.to_csv(bad_path, index=False)

    # Summary statistics
    print("\n" + "=" * 60)
    print("CLEAN DATA SUMMARY (demonstrates realistic distributions)")
    print("=" * 60)
    print(clean_df[["weight_kg", "height_cm", "age"]].describe().round(1))

    print("\n" + "=" * 60)
    print("SAMPLE RECORDS (showing Faker-generated names)")
    print("=" * 60)
    print(clean_df[["patient_id", "first_name", "last_name", "weight_kg",
                    "height_cm", "age", "sex"]].head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("CONFIG-DRIVEN DEVELOPMENT DEMO")
    print("=" * 60)
    print(f"All parameters loaded from: {config_path.name}")
    print(f"  - Sample sizes: {config['sample_sizes']}")
    print(f"  - Random seed: {config['random_seed']} (reproducible!)")
    print(f"  - Violations: {len(config['violations'])} boundary violations configured")
    print("\nTo change distributions, edit the YAML config—no code changes needed!")


if __name__ == "__main__":
    main()
