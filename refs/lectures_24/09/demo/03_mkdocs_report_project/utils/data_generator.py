#!/usr/bin/env python3
"""
Data generator for neonatal feeding study.
Creates synthetic data that mimics real clinical data patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def generate_synthetic_data(n_patients=100, seed=42):
    """
    Generate synthetic neonatal feeding study data.

    Args:
        n_patients (int): Number of patients to generate
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Synthetic patient data
    """
    np.random.seed(seed)

    # Generate base characteristics
    data = {
        "gestational_age": np.random.normal(32, 2, n_patients),  # weeks
        "birth_weight": np.random.normal(1800, 400, n_patients),  # grams
        "ventilation_status": np.random.choice(
            ["Yes", "No"], n_patients, p=[0.3, 0.7]
        ),
        "apgar_5min": np.random.randint(5, 10, n_patients),
        "maternal_diabetes": np.random.choice(
            ["Yes", "No"], n_patients, p=[0.15, 0.85]
        ),
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Generate time to full oral feeding based on characteristics
    base_time = 30  # Base time in days

    # Effect of gestational age (negative correlation)
    ga_effect = -2.5 * (df["gestational_age"] - 32)

    # Effect of ventilation (positive)
    vent_effect = np.where(df["ventilation_status"] == "Yes", 10, 0)

    # Effect of birth weight (negative correlation)
    weight_effect = -0.01 * (df["birth_weight"] - 1800)

    # Effect of Apgar (negative correlation)
    apgar_effect = -1.5 * (df["apgar_5min"] - 7)

    # Effect of maternal diabetes (positive)
    diabetes_effect = np.where(df["maternal_diabetes"] == "Yes", 5, 0)

    # Add some random noise
    noise = np.random.normal(0, 5, n_patients)

    # Calculate final time to full oral feeding
    df["time_to_fof"] = (
        base_time
        + ga_effect
        + vent_effect
        + weight_effect
        + apgar_effect
        + diabetes_effect
        + noise
    )

    # Ensure no negative times
    df["time_to_fof"] = df["time_to_fof"].clip(lower=1)

    # Round to whole days
    df["time_to_fof"] = df["time_to_fof"].round()

    # Save data dictionary
    data_dict = {
        "gestational_age": {
            "display_name": "Gestational Age",
            "type": "continuous",
            "units": "weeks",
            "description": "Gestational age at birth",
        },
        "birth_weight": {
            "display_name": "Birth Weight",
            "type": "continuous",
            "units": "grams",
            "description": "Birth weight in grams",
        },
        "ventilation_status": {
            "display_name": "Mechanical Ventilation",
            "type": "categorical",
            "description": "Whether the infant required mechanical ventilation",
        },
        "apgar_5min": {
            "display_name": "5-minute Apgar Score",
            "type": "ordinal",
            "description": "Apgar score at 5 minutes of life",
        },
        "maternal_diabetes": {
            "display_name": "Maternal Diabetes",
            "type": "categorical",
            "description": "Whether the mother had diabetes",
        },
        "time_to_fof": {
            "display_name": "Time to Full Oral Feeding",
            "type": "continuous",
            "units": "days",
            "description": "Days until full oral feeding achieved",
        },
    }

    # Save data and dictionary
    output_dir = Path(__file__).parent.parent / "docs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "synthetic_data.csv", index=False)
    with open(output_dir / "data_dictionary.json", "w") as f:
        json.dump(data_dict, f, indent=2)

    return df
