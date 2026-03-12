#!/usr/bin/env python3
"""
Visualization module for neonatal feeding study.
Provides generic chart generation methods that can be customized by reports.
"""

import altair as alt
import pandas as pd
import numpy as np
from pathlib import Path
from utils import report_generator

# Configure Altair
alt.data_transformers.enable("json")


def generate_histogram(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    color: str = "steelblue",
    opacity: float = 0.7,
    width: int = 280,
    height: int = 200,
    maxbins: int = 12,
) -> alt.Chart:
    """
    Generate a histogram chart.

    Args:
        df: Data to plot
        x_col: Column to plot on x-axis
        title: Chart title
        color: Bar color
        opacity: Bar opacity
        width: Chart width
        height: Chart height
        maxbins: Maximum number of bins

    Returns:
        alt.Chart: Altair chart object
    """
    return (
        alt.Chart(df)
        .mark_bar(opacity=opacity, color=color)
        .encode(
            alt.X(
                f"{x_col}:Q",
                bin=alt.Bin(maxbins=maxbins),
                title=x_col.replace("_", " ").title(),
            ),
            alt.Y("count()", title="Count"),
            tooltip=["count()"],
        )
        .properties(width=width, height=height, title=title)
    )


def generate_demographics_overview(
    df: pd.DataFrame,
    ga_col: str = "gestational_age",
    weight_col: str = "birth_weight",
) -> alt.Chart:
    """
    Generate a demographics overview chart combining gestational age and birth weight histograms.

    Args:
        df: Data to plot
        ga_col: Column name for gestational age
        weight_col: Column name for birth weight

    Returns:
        alt.Chart: Combined Altair chart object
    """
    ga_hist = generate_histogram(
        df,
        x_col=ga_col,
        title="Gestational Age Distribution",
        color="steelblue",
    )
    weight_hist = generate_histogram(
        df,
        x_col=weight_col,
        title="Birth Weight Distribution",
        color="orange",
    )

    return (
        alt.hconcat(ga_hist, weight_hist)
        .resolve_scale(x="independent", y="independent")
        .properties(title="Patient Demographics Overview")
        .configure_view(
            continuousWidth=300, continuousHeight=300, strokeWidth=0
        )
    )


def generate_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = None,
    size_col: str = None,
    title: str = None,
    x_domain: tuple = None,
    y_domain: tuple = None,
    width: int = 600,
    height: int = 400,
    add_regression: bool = False,
) -> alt.Chart:
    """
    Generate a scatter plot with optional regression line.

    Args:
        df: Data to plot
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Column for point color
        size_col: Column for point size
        title: Chart title
        x_domain: Tuple of (min, max) for x-axis
        y_domain: Tuple of (min, max) for y-axis
        width: Chart width
        height: Chart height
        add_regression: Whether to add regression line

    Returns:
        alt.Chart: Altair chart object
    """
    # Create base chart with zoom/pan
    base = alt.Chart(df).add_selection(
        alt.selection_interval(bind="scales", zoom=False)
    )

    # Build encoding
    encoding = {
        "x": alt.X(
            f"{x_col}:Q",
            title=x_col.replace("_", " ").title(),
            scale=alt.Scale(domain=x_domain) if x_domain else None,
        ),
        "y": alt.Y(
            f"{y_col}:Q",
            title=y_col.replace("_", " ").title(),
            scale=alt.Scale(domain=y_domain) if y_domain else None,
        ),
    }

    if color_col:
        encoding["color"] = alt.Color(
            f"{color_col}:N",
            title=color_col.replace("_", " ").title(),
        )
    if size_col:
        encoding["size"] = alt.Size(
            f"{size_col}:Q",
            title=size_col.replace("_", " ").title(),
            scale=alt.Scale(range=[50, 200]),
        )

    # Add tooltips
    tooltip = [
        f"{col}:Q" if df[col].dtype.kind in "ifc" else f"{col}:N"
        for col in [x_col, y_col, color_col, size_col]
        if col
    ]
    encoding["tooltip"] = tooltip

    # Create scatter plot
    scatter = base.mark_circle(size=80, opacity=0.7).encode(**encoding)

    # Add regression line if requested
    if add_regression:
        regression = (
            base.mark_line(color="red", strokeWidth=2)
            .transform_regression(x_col, y_col)
            .encode(alt.X(f"{x_col}:Q"), alt.Y(f"{y_col}:Q"))
        )
        chart = (scatter + regression).properties(
            width=width, height=height, title=title
        )
    else:
        chart = scatter.properties(width=width, height=height, title=title)

    return chart


def generate_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    width: int = 600,
    height: int = 400,
) -> alt.Chart:
    """
    Generate a correlation matrix heatmap.

    Args:
        corr_matrix: Correlation matrix as DataFrame
        title: Chart title
        width: Chart width
        height: Chart height

    Returns:
        alt.Chart: Altair chart object
    """
    # Prepare data for heatmap
    corr_data = corr_matrix.stack().reset_index()
    corr_data.columns = ["variable1", "variable2", "correlation"]

    # Create heatmap
    return (
        alt.Chart(corr_data)
        .mark_rect()
        .encode(
            alt.X("variable1:N", title=""),
            alt.Y("variable2:N", title=""),
            alt.Color(
                "correlation:Q",
                scale=alt.Scale(
                    domain=[-1, 1],
                    range=["red", "white", "blue"],
                ),
                title="Correlation",
            ),
            tooltip=["variable1:N", "variable2:N", "correlation:Q"],
        )
        .properties(width=width, height=height, title=title)
    )


def generate_box_plots(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    width: int = 400,
    height: int = 300,
) -> alt.Chart:
    """
    Generate box plots for categorical variable analysis.

    Args:
        df: Data to plot
        x_col: Categorical column for x-axis
        y_col: Numeric column for y-axis
        title: Chart title
        width: Chart width
        height: Chart height

    Returns:
        alt.Chart: Altair chart object
    """
    return (
        alt.Chart(df)
        .mark_boxplot()
        .encode(
            alt.X(f"{x_col}:N", title=x_col.replace("_", " ").title()),
            alt.Y(f"{y_col}:Q", title=y_col.replace("_", " ").title()),
            tooltip=[
                f"{x_col}:N",
                alt.Tooltip(f"{y_col}:Q", format=".1f"),
            ],
        )
        .properties(width=width, height=height, title=title)
    )


def generate_grouped_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    width: int = 120,
    height: int = 250,
) -> alt.Chart:
    """
    Generate a grouped bar chart.

    Args:
        df: Data to plot
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Column for bar color
        title: Chart title
        width: Chart width
        height: Chart height

    Returns:
        alt.Chart: Altair chart object
    """
    return (
        alt.Chart(df)
        .mark_bar(opacity=0.8)
        .encode(
            alt.X(f"{x_col}:N", title=x_col.replace("_", " ").title()),
            alt.Y(f"{y_col}:Q", title=y_col.replace("_", " ").title()),
            alt.Color(
                f"{color_col}:N", title=color_col.replace("_", " ").title()
            ),
            alt.Column(f"{x_col}:N", title=""),
            tooltip=[
                f"{x_col}:N",
                f"{color_col}:N",
                alt.Tooltip(f"{y_col}:Q", format=".1f"),
                "count:Q",
            ],
        )
        .properties(width=width, height=height, title=title)
        .resolve_scale(x="independent")
    )


def save_chart(chart: alt.Chart, output_path: Path) -> str:
    """
    Save an Altair chart as JSON.

    Args:
        chart: Altair chart to save
        output_path: Path to save chart

    Returns:
        str: Path to saved chart
    """
    report_generator.save_altair_chart_json(chart, output_path)
    return str(output_path)
