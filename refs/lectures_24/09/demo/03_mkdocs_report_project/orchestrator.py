#!/usr/bin/env python3
"""
Orchestrator script for Optimal Oral Feeding Time analysis.
This script coordinates the analysis pipeline and documentation generation.
"""

import sys
import os
import shutil
import pandas as pd
import json
from pathlib import Path

from utils.reports import bivariate_analysis
from utils.reports.rpt_10_interactive_visualizations import (
    interactive_visualizations_report,
)
from utils import report_generator
from utils import data_generator

DATA_FILE = os.path.join(
    os.path.dirname(__file__), "data", "3.18.25_cohort.tsv"
)
RESULTS_DIR = "results"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
REPORTS_MEDIA_DIR = os.path.join(REPORTS_DIR, "media")

if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR)
os.makedirs(REPORTS_DIR)
os.makedirs(REPORTS_MEDIA_DIR)

readme_path = "README.md"
index_path = os.path.join(RESULTS_DIR, "index.md")
if os.path.exists(readme_path):
    shutil.copy2(readme_path, index_path)


def run_pipeline():
    # Load or generate data
    if os.path.exists(DATA_FILE):
        print(f"Loading data from {DATA_FILE}")
        data = pd.read_csv(DATA_FILE, sep="\t")
    else:
        print("Data file not found. Generating synthetic data...")
        data = data_generator.generate_synthetic_data(n_patients=100)
        print(f"Generated {len(data)} synthetic patient records")

    # Generate interactive visualizations report
    viz_report = interactive_visualizations_report(data, REPORTS_MEDIA_DIR)
    viz_report_path = report_generator.generate_report(
        viz_report["sections"],
        output_path=os.path.join(RESULTS_DIR, viz_report["filename"]),
        title=viz_report["title"],
    )

    # Generate bivariate analysis report
    bivar_report = bivariate_analysis.generate_bivariate_report(
        data, RESULTS_DIR, REPORTS_MEDIA_DIR
    )
    bivar_report_path = report_generator.generate_report(
        bivar_report["sections"],
        output_path=os.path.join(RESULTS_DIR, bivar_report["filename"]),
        title=bivar_report["title"],
    )

    print(f"Interactive Visualizations report: {viz_report_path}")
    print(f"Bivariate Analysis report: {bivar_report_path}")


if __name__ == "__main__":
    run_pipeline()
