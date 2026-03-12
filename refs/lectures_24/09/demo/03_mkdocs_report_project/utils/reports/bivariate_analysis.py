#!/usr/bin/env python3
"""
Bivariate analysis report for neonatal feeding study.
Generates statistical visualizations and creates a report with embedded charts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils import visualization
from utils import report_generator
import scipy.stats


def generate_bivariate_report(
    data: pd.DataFrame, output_dir: str, media_dir: str
) -> dict:
    """
    Generate a bivariate analysis report for the neonatal feeding study.

    Args:
        data (pd.DataFrame): Patient data to analyze
        output_dir (str): Directory to save the report
        media_dir (str): Directory to save charts

    Returns:
        dict: Report object containing sections and metadata
    """
    # Generate correlation matrix
    numeric_cols = ["gestational_age", "birth_weight", "time_to_fof"]
    corr_matrix = data[numeric_cols].corr()
    corr_chart = visualization.generate_correlation_heatmap(
        corr_matrix,
        title="Correlation Matrix of Key Variables",
    )
    corr_path = visualization.save_chart(
        corr_chart, Path(media_dir) / "correlation_matrix.json"
    )

    # Generate box plots for categorical variables
    box_chart = visualization.generate_box_plots(
        data,
        x_col="ventilation_status",
        y_col="time_to_fof",
        title="Time to Full Oral Feeding by Ventilation Status",
    )
    box_path = visualization.save_chart(
        box_chart, Path(media_dir) / "ventilation_box_plot.json"
    )

    # Create statistical summary table
    stats_summary = []
    for col in ["gestational_age", "birth_weight"]:
        corr = data[col].corr(data["time_to_fof"])
        # Calculate p-value using t-test
        n = len(data)
        t = corr * np.sqrt((n - 2) / (1 - corr**2))
        p_value = 2 * (1 - scipy.stats.t.cdf(abs(t), n - 2))
        stats_summary.append({
            "Variable": col.replace("_", " ").title(),
            "Correlation": f"{corr:.3f}",
            "P-value": f"{p_value:.3f}",
        })
    stats_df = pd.DataFrame(stats_summary)

    # Create report sections with relative paths
    sections = [
        (
            "Bivariate Analysis",
            """This report presents a statistical analysis of relationships between
            key variables in our neonatal feeding study. We examine both numeric and
            categorical variables to understand their impact on time to full oral feeding.""",
            None,
            None,
            None,
        ),
        (
            "Correlation Analysis",
            """The correlation matrix below shows the relationships between our key
            numeric variables. We can see strong correlations between gestational age,
            birth weight, and time to full oral feeding.""",
            None,
            f"media/{Path(corr_path).name}",
            None,
        ),
        (
            "Categorical Variable Analysis",
            """The box plots below show the distribution of time to full oral feeding
            for different categories of our categorical variables. This helps us understand
            how these factors affect feeding progression.""",
            None,
            f"media/{Path(box_path).name}",
            None,
        ),
        (
            "Statistical Summary",
            """The table below provides a statistical summary of the relationships
            between our key variables and time to full oral feeding. We include both
            correlation coefficients and p-values to assess the strength and significance
            of these relationships.""",
            stats_df.to_markdown(index=False),
            None,
            None,
        ),
    ]

    return {
        "sections": sections,
        "title": "Bivariate Analysis",
        "filename": "reports/bivariate_analysis.md",
    }
