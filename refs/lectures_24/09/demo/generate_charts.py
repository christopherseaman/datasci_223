#!/usr/bin/env python3
"""
Generate Altair chart specifications for the lecture examples.
This script creates JSON files for various Altair charts used in the lecture.
"""

import os
from pathlib import Path
import altair as alt
import pandas as pd
import numpy as np

# Create media directory if it doesn't exist
media_dir = Path("media")
media_dir.mkdir(exist_ok=True)

# Generate sample data
np.random.seed(42)
df = pd.DataFrame({
    "x": np.random.normal(0, 1, 100),
    "y": np.random.normal(0, 1, 100),
    "category": np.random.choice(["A", "B", "C"], 100),
})


def save_chart(chart, filename):
    """Save chart specification to JSON file."""
    chart.save(str(media_dir / f"{filename}.json"))


# 1. Scatter Plot with Marginal Histograms
scatter = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x="x:Q",
        y="y:Q",
        color="category:N",
        tooltip=["x:Q", "y:Q", "category:N"],
    )
    .properties(width=400, height=400)
)

x_hist = (
    alt.Chart(df)
    .mark_bar()
    .encode(x=alt.X("x:Q", bin=True), y="count()")
    .properties(width=400, height=100)
)

y_hist = (
    alt.Chart(df)
    .mark_bar()
    .encode(y=alt.Y("y:Q", bin=True), x="count()")
    .properties(width=100, height=400)
)

marginal_chart = x_hist & (scatter | y_hist)
save_chart(marginal_chart, "chart_marginal_histograms")

# 2. Interactive Variable Selection
var_select = alt.param(
    name="var_select",
    bind=alt.binding_select(
        options=["x", "y", "category"], name="Select Variable: "
    ),
    value="x",
)

var_selection_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x=alt.X("x:Q"),
        y=alt.Y("y:Q"),
        color=alt.condition(
            var_select == "category", "category:N", alt.value("steelblue")
        ),
    )
    .add_params(var_select)
)
save_chart(var_selection_chart, "chart_variable_selection")

# 3. Layered Chart with Multiple Marks
base = alt.Chart(df).encode(x="x:Q", y="y:Q")

points = base.mark_circle().encode(
    color="category:N", tooltip=["x:Q", "y:Q", "category:N"]
)

trend = base.mark_line(color="red").transform_regression("x", "y")

layered_chart = (points + trend).properties(
    width=400, height=300, title="Scatter Plot with Trend Line"
)
save_chart(layered_chart, "chart_layered")

# 4. Interactive Brushing and Linking
brush = alt.selection_interval()

chart1 = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x="x:Q",
        y="y:Q",
        color=alt.condition(brush, "category:N", alt.value("lightgray")),
        tooltip=["x:Q", "y:Q", "category:N"],
    )
    .add_params(brush)
    .properties(width=300, height=300)
)

chart2 = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x="category:N",
        y="y:Q",
        color=alt.condition(brush, "category:N", alt.value("lightgray")),
        tooltip=["x:Q", "y:Q", "category:N"],
    )
    .add_params(brush)
    .properties(width=300, height=300)
)

brushing_chart = chart1 | chart2
save_chart(brushing_chart, "chart_brushing")

print("Chart specifications generated successfully!")
