import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import numpy as np

# Generate synthetic health data
np.random.seed(42)
n_samples = 100

df = pd.DataFrame({
    "patient_id": range(1, n_samples + 1),
    "age": np.random.normal(45, 15, n_samples).astype(int),
    "blood_pressure": np.random.normal(120, 10, n_samples).astype(int),
    "heart_rate": np.random.normal(75, 8, n_samples).astype(int),
    "condition": np.random.choice(
        ["Healthy", "Hypertension", "Arrhythmia"], n_samples
    ),
})

# Initialize the Dash app
app = dash.Dash(__name__)

# Create the layout
app.layout = html.Div([
    html.H1("Patient Health Dashboard", style={"textAlign": "center"}),
    # Filters
    html.Div(
        [
            html.Label("Select Condition:"),
            dcc.Dropdown(
                id="condition-filter",
                options=[
                    {"label": x, "value": x} for x in df["condition"].unique()
                ],
                value="Healthy",
            ),
        ],
        style={"width": "30%", "margin": "20px"},
    ),
    # Charts
    html.Div(
        [
            dcc.Graph(
                id="age-distribution",
                figure=px.histogram(
                    df[df["condition"] == "Healthy"],
                    x="age",
                    title="Age Distribution",
                    nbins=20,
                ),
            ),
            dcc.Graph(
                id="vitals-scatter",
                figure=px.scatter(
                    df[df["condition"] == "Healthy"],
                    x="blood_pressure",
                    y="heart_rate",
                    title="Blood Pressure vs Heart Rate",
                    color="age",
                    size="age",
                ),
            ),
        ],
        style={"display": "flex", "flexWrap": "wrap"},
    ),
])


# Callback to update charts based on filter
@app.callback(
    [
        dash.Output("age-distribution", "figure"),
        dash.Output("vitals-scatter", "figure"),
    ],
    [dash.Input("condition-filter", "value")],
)
def update_charts(selected_condition):
    filtered_df = df[df["condition"] == selected_condition]

    age_fig = px.histogram(
        filtered_df,
        x="age",
        title=f"Age Distribution - {selected_condition}",
        nbins=20,
    )

    vitals_fig = px.scatter(
        filtered_df,
        x="blood_pressure",
        y="heart_rate",
        title=f"Blood Pressure vs Heart Rate - {selected_condition}",
        color="age",
        size="age",
    )

    return age_fig, vitals_fig


if __name__ == "__main__":
    app.run(debug=True)
