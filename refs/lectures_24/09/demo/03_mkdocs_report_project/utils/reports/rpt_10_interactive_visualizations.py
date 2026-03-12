#!/usr/bin/env python3
"""
Interactive visualizations report for neonatal feeding study.
Generates interactive charts and creates a report with embedded visualizations.
"""

import pandas as pd
from pathlib import Path
from utils import visualization
from utils import report_generator


def interactive_visualizations_report(
    data: pd.DataFrame, media_dir: str
) -> dict:
    """
    Generate a report with interactive visualizations.

    Args:
        data (pd.DataFrame): Patient data to visualize
        media_dir (str): Directory to save charts

    Returns:
        dict: Report object containing sections and metadata
    """
    # Generate demographics overview chart
    demographics_chart = visualization.generate_demographics_overview(data)
    demographics_path = visualization.save_chart(
        demographics_chart, Path(media_dir) / "demographics_overview.json"
    )

    # Generate primary analysis chart
    correlation = data["gestational_age"].corr(data["time_to_fof"])
    primary_chart = visualization.generate_scatter_plot(
        data,
        x_col="gestational_age",
        y_col="time_to_fof",
        color_col="ventilation_status",
        size_col="birth_weight",
        title=f"Gestational Age vs Time to Full Oral Feeding (r = {correlation:.3f})",
        x_domain=(24, 32),
        y_domain=(0, 45),
        add_regression=True,
    )
    primary_path = visualization.save_chart(
        primary_chart, Path(media_dir) / "primary_analysis.json"
    )

    # Generate clinical factors chart
    clinical_summary = []
    for factor, col in [
        ("Mechanical Ventilation", "ventilation_status"),
        ("Maternal Diabetes", "maternal_diabetes"),
    ]:
        summary = (
            data.groupby(col)["time_to_fof"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        for _, row in summary.iterrows():
            clinical_summary.append({
                "factor": factor,
                "category": row[col],
                "mean_time_to_fof": row["mean"],
                "std_time_to_fof": row["std"],
                "count": row["count"],
            })
    summary_df = pd.DataFrame(clinical_summary)
    clinical_chart = visualization.generate_grouped_bar_chart(
        summary_df,
        x_col="factor",
        y_col="mean_time_to_fof",
        color_col="category",
        title="Clinical Factors Impact on Feeding Progression",
    )
    clinical_path = visualization.save_chart(
        clinical_chart, Path(media_dir) / "clinical_factors.json"
    )

    # Create report sections with relative paths
    sections = [
        (
            "Interactive Visualizations",
            """This page contains interactive visualizations of the neonatal feeding study data.
            You can hover over data points to see details, zoom in/out, and pan around the charts.
            The visualizations are powered by Vega-Lite and are fully interactive.""",
            None,
            None,
            None,
        ),
        (
            "Patient Demographics",
            """The following charts show the distribution of key demographic variables
            in our study population. These include gestational age and birth weight distributions.""",
            None,
            f"media/{Path(demographics_path).name}",
            None,
        ),
        (
            "Primary Analysis: Gestational Age Impact",
            """This interactive scatter plot shows the relationship between gestational age
            and time to full oral feeding. The size of each point represents birth weight, and color
            indicates mechanical ventilation status. The red line shows the linear regression fit.""",
            None,
            f"media/{Path(primary_path).name}",
            None,
        ),
        (
            "Clinical Factors Impact",
            """These grouped bar charts show how different clinical factors affect the
            time to full oral feeding. Each chart shows the mean time to full oral feeding for
            different categories of each factor.""",
            None,
            f"media/{Path(clinical_path).name}",
            None,
        ),
    ]

    return {
        "sections": sections,
        "title": "Interactive Visualizations",
        "filename": "reports/visualization.md",
    }
