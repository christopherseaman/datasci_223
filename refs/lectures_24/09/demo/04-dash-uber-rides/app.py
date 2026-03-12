# =============================================
# Dash Uber Rides App (Annotated for Teaching)
# =============================================
# This script creates an interactive dashboard for exploring Uber ride data in New York City.
# It demonstrates Dash layout, callbacks, data wrangling, and interactive mapping.
# Comments are provided for teaching and self-study. No code or formatting is changed.

# --- Imports ---
import dash  # Dash: main web app framework
from dash import dcc, html  # Dash components: graphs, dropdowns, etc.
import pandas as pd  # Data wrangling
import numpy as np  # Numeric operations

from dash.dependencies import Input, Output  # For callback wiring
from plotly import graph_objs as go  # Plotly graph objects
from plotly.graph_objs import *  # (Wildcard import for brevity)
from datetime import datetime as dt  # Date/time handling

# --- Dash App Initialization ---
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "New York Uber Rides"
server = app.server  # For deployment (e.g., with gunicorn)

# --- Mapbox Token (for map backgrounds) ---
# Get your free token at https://account.mapbox.com/access-tokens/
# Create a token with the 'styles:read' scope
mapbox_access_token = "YOUR_TOKEN_HERE"  # Replace with your token

# --- Error if token not set ---
if mapbox_access_token == "YOUR_TOKEN_HERE":
    raise RuntimeError(
        "Mapbox token not set. Please get a free token at https://account.mapbox.com/access-tokens/ and update the mapbox_access_token variable in app.py."
    )

# --- Key NYC Landmarks for Map Centering ---
list_of_locations = {
    "Madison Square Garden": {"lat": 40.7505, "lon": -73.9934},
    "Yankee Stadium": {"lat": 40.8296, "lon": -73.9262},
    "Empire State Building": {"lat": 40.7484, "lon": -73.9857},
    "New York Stock Exchange": {"lat": 40.7069, "lon": -74.0113},
    "JFK Airport": {"lat": 40.644987, "lon": -73.785607},
    "Grand Central Station": {"lat": 40.7527, "lon": -73.9772},
    "Times Square": {"lat": 40.7589, "lon": -73.9851},
    "Columbia University": {"lat": 40.8075, "lon": -73.9626},
    "United Nations HQ": {"lat": 40.7489, "lon": -73.9680},
}

# --- Data Loading and Preprocessing ---
# The app uses three months of Uber ride data from FiveThirtyEight's public dataset.
# Data is loaded directly from GitHub for convenience.
df1 = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data1.csv",
    dtype=object,
)
df2 = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data2.csv",
    dtype=object,
)
df3 = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data3.csv",
    dtype=object,
)
df = pd.concat([df1, df2, df3], axis=0)
df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%Y-%m-%d %H:%M:%S")
df.index = df["Date/Time"]  # Set the index for easy time slicing
df.drop(
    columns=["Date/Time"], inplace=True
)  # Updated drop syntax for newer pandas

# --- Precompute a nested list of DataFrames for fast access by month and day ---
totalList = []
for month in df.groupby(df.index.month):
    dailyList = []
    for day in month[1].groupby(month[1].index.day):
        dailyList.append(day[1])
    totalList.append(dailyList)

# --- Layout Definition ---
# The app uses a two-column layout: controls on the left, visualizations on the right.
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # --- User Controls Column ---
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.A(
                            html.Img(
                                className="logo",
                                src=app.get_asset_url("dash-logo-new.png"),
                            ),
                            href="https://plotly.com/dash/",
                        ),
                        html.H2("DASH - UBER DATA APP"),
                        html.P(
                            """Select different days using the date picker or by selecting
                            different time frames on the histogram."""
                        ),
                        # --- Date Picker ---
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                dcc.DatePickerSingle(
                                    id="date-picker",
                                    min_date_allowed=dt(2014, 4, 1),
                                    max_date_allowed=dt(2014, 9, 30),
                                    initial_visible_month=dt(2014, 4, 1),
                                    date=dt(2014, 4, 1).date(),
                                    display_format="MMMM D, YYYY",
                                    style={"border": "0px solid black"},
                                )
                            ],
                        ),
                        # --- Dropdowns for Location and Hour Selection ---
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Location dropdown (centers map on landmark)
                                        dcc.Dropdown(
                                            id="location-dropdown",
                                            options=[
                                                {"label": i, "value": i}
                                                for i in list_of_locations
                                            ],
                                            placeholder="Select a location",
                                        )
                                    ],
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Hour selection dropdown (filters rides by hour)
                                        dcc.Dropdown(
                                            id="bar-selector",
                                            options=[
                                                {
                                                    "label": str(n) + ":00",
                                                    "value": str(n),
                                                }
                                                for n in range(24)
                                            ],
                                            multi=True,
                                            placeholder="Select certain hours",
                                        )
                                    ],
                                ),
                            ],
                        ),
                        # --- Dynamic Text Outputs ---
                        html.P(id="total-rides"),
                        html.P(id="total-rides-selection"),
                        html.P(id="date-value"),
                        # --- Data Source and Links ---
                        dcc.Markdown(
                            """
                            Source: [FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response/tree/master/uber-trip-data)

                            Links: [Source Code](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-uber-rides-demo) | [Enterprise Demo](https://plotly.com/get-demo/)
                            """
                        ),
                    ],
                ),
                # --- Visualization Column ---
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id="map-graph"),  # Map of ride locations
                        html.Div(
                            className="text-padding",
                            children=[
                                "Select any of the bars on the histogram to section data by time."
                            ],
                        ),
                        dcc.Graph(id="histogram"),  # Histogram of rides by hour
                    ],
                ),
            ],
        )
    ]
)

# --- Helper Data Structures ---
# Number of days in each month (April-September 2014)
daysInMonth = [30, 31, 30, 31, 31, 30]
# Month index for display
monthIndex = pd.Index(["Apr", "May", "June", "July", "Aug", "Sept"])


# --- Helper Function: get_selection ---
# Returns arrays for x (hour), y (ride count), and color for the histogram.
# Highlights selected hours in white.
def get_selection(month, day, selection):
    xVal = []
    yVal = []
    xSelected = []
    colorVal = [
        # Color palette for each hour (24 colors)
        "#F4EC15",
        "#DAF017",
        "#BBEC19",
        "#9DE81B",
        "#80E41D",
        "#66E01F",
        "#4CDC20",
        "#34D822",
        "#24D249",
        "#25D042",
        "#26CC58",
        "#28C86D",
        "#29C481",
        "#2AC093",
        "#2BBCA4",
        "#2BB5B8",
        "#2C99B4",
        "#2D7EB0",
        "#2D65AC",
        "#2E4EA4",
        "#2E38A4",
        "#3B2FA0",
        "#4E2F9C",
        "#603099",
    ]
    # Convert selected hours (from dropdown or histogram) to integers
    xSelected.extend([int(x) for x in selection])
    for i in range(24):  # Loop over each hour of the day
        # If this hour is selected, color it white for emphasis
        if i in xSelected and len(xSelected) < 24:
            colorVal[i] = "#FFFFFF"
        xVal.append(i)  # Add hour to x-axis
        # Count rides for this hour (using pandas time indexing)
        yVal.append(
            len(totalList[month][day][totalList[month][day].index.hour == i])
        )
    # Return arrays for plotting
    return [np.array(xVal), np.array(yVal), np.array(colorVal)]


# --- Callbacks ---
# (Callbacks are functions that update the UI in response to user input)


# 1. Histogram bar selection updates the hour dropdown
@app.callback(
    Output("bar-selector", "value"),
    [Input("histogram", "selectedData"), Input("histogram", "clickData")],
)
def update_bar_selector(value, clickData):
    holder = []
    if clickData:
        holder.append(str(int(clickData["points"][0]["x"])))
    if value:
        for x in value["points"]:
            holder.append(str(int(x["x"])))
    return list(set(holder))


# 2. Clear histogram selection if a bar is clicked
@app.callback(
    Output("histogram", "selectedData"), [Input("histogram", "clickData")]
)
def update_selected_data(clickData):
    if clickData:
        return {"points": []}


# 3. Update total rides for the selected day
@app.callback(Output("total-rides", "children"), [Input("date-picker", "date")])
def update_total_rides(datePicked):
    date_picked = dt.strptime(
        datePicked, "%Y-%m-%d"
    )  # Convert string to datetime
    return "Total Number of rides: {:,d}".format(
        len(totalList[date_picked.month - 4][date_picked.day - 1])
    )


# 4. Update total rides for selected hours (and show which hours)
@app.callback(
    [
        Output("total-rides-selection", "children"),
        Output("date-value", "children"),
    ],
    [Input("date-picker", "date"), Input("bar-selector", "value")],
)
def update_total_rides_selection(datePicked, selection):
    firstOutput = ""
    if selection is not None and len(selection) != 0:
        date_picked = dt.strptime(datePicked, "%Y-%m-%d")
        totalInSelection = 0
        for x in selection:
            totalInSelection += len(
                totalList[date_picked.month - 4][date_picked.day - 1][
                    totalList[date_picked.month - 4][
                        date_picked.day - 1
                    ].index.hour
                    == int(x)
                ]
            )
        firstOutput = "Total rides in selection: {:,d}".format(totalInSelection)
    if (
        datePicked is None
        or selection is None
        or len(selection) == 24
        or len(selection) == 0
    ):
        return firstOutput, (datePicked, " - showing hour(s): All")
    holder = sorted([int(x) for x in selection])
    if holder == list(range(min(holder), max(holder) + 1)):
        return (
            firstOutput,
            (
                datePicked,
                " - showing hour(s): ",
                holder[0],
                "-",
                holder[len(holder) - 1],
            ),
        )
    holder_to_string = ", ".join(str(x) for x in holder)
    return firstOutput, (datePicked, " - showing hour(s): ", holder_to_string)


# 5. Update histogram figure based on date and hour selection
@app.callback(
    Output("histogram", "figure"),
    [Input("date-picker", "date"), Input("bar-selector", "value")],
)
def update_histogram(datePicked, selection):
    date_picked = dt.strptime(datePicked, "%Y-%m-%d")
    monthPicked = date_picked.month - 4
    dayPicked = date_picked.day - 1
    [xVal, yVal, colorVal] = get_selection(monthPicked, dayPicked, selection)
    layout = go.Layout(
        bargap=0.01,  # Small gap between bars
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=50),
        showlegend=False,
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
        dragmode="select",  # Enable box/lasso selection
        font=dict(color="white"),
        xaxis=dict(
            range=[-0.5, 23.5],
            showgrid=False,
            nticks=25,
            fixedrange=True,
            ticksuffix=":00",
        ),
        yaxis=dict(
            range=[0, max(yVal) + max(yVal) / 4],
            showticklabels=False,
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        # Annotate each bar with its count
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )
    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(color=colorVal), hoverinfo="x"),
            # Invisible scatter for better selection UX
            go.Scatter(
                opacity=0,
                x=xVal,
                y=yVal / 2,
                hoverinfo="none",
                mode="markers",
                marker=dict(
                    color="rgb(66, 134, 244, 0)", symbol="square", size=40
                ),
                visible=True,
            ),
        ],
        layout=layout,
    )


# --- Helper: Get ride coordinates for selected hours ---
def getLatLonColor(selectedData, month, day):
    listCoords = totalList[month][day]
    # If no hours are selected, return all rides for the day
    if selectedData is None or len(selectedData) == 0:
        return listCoords
    # Build a boolean mask for the selected hours
    listStr = "listCoords["
    for time in selectedData:
        if selectedData.index(time) is not len(selectedData) - 1:
            listStr += (
                "(totalList[month][day].index.hour==" + str(int(time)) + ") | "
            )
        else:
            listStr += (
                "(totalList[month][day].index.hour==" + str(int(time)) + ")]"
            )
    # Use eval to filter DataFrame (not best practice, but works for teaching)
    return eval(listStr)


# 6. Update the map graph based on date, hour, and location selection
@app.callback(
    Output("map-graph", "figure"),
    [
        Input("date-picker", "date"),
        Input("bar-selector", "value"),
        Input("location-dropdown", "value"),
    ],
)
def update_graph(datePicked, selectedData, selectedLocation):
    zoom = 12.0
    latInitial = 40.7272
    lonInitial = -73.991251
    bearing = 0
    # If a landmark is selected, center map on it and zoom in
    if selectedLocation:
        zoom = 15.0
        latInitial = list_of_locations[selectedLocation]["lat"]
        lonInitial = list_of_locations[selectedLocation]["lon"]
    date_picked = dt.strptime(datePicked, "%Y-%m-%d")
    monthPicked = date_picked.month - 4
    dayPicked = date_picked.day - 1
    listCoords = getLatLonColor(selectedData, monthPicked, dayPicked)
    return go.Figure(
        data=[
            # Plot all ride locations as points, colored by hour
            Scattermapbox(
                lat=listCoords["Lat"],
                lon=listCoords["Lon"],
                mode="markers",
                hoverinfo="lat+lon+text",
                text=listCoords.index.hour,  # Show hour on hover
                marker=dict(
                    showscale=True,
                    color=np.append(np.insert(listCoords.index.hour, 0, 0), 23),
                    opacity=0.5,
                    size=5,
                    colorscale=[
                        [0, "#F4EC15"],
                        [0.04167, "#DAF017"],
                        [0.0833, "#BBEC19"],
                        [0.125, "#9DE81B"],
                        [0.1667, "#80E41D"],
                        [0.2083, "#66E01F"],
                        [0.25, "#4CDC20"],
                        [0.292, "#34D822"],
                        [0.333, "#24D249"],
                        [0.375, "#25D042"],
                        [0.4167, "#26CC58"],
                        [0.4583, "#28C86D"],
                        [0.50, "#29C481"],
                        [0.54167, "#2AC093"],
                        [0.5833, "#2BBCA4"],
                        [1.0, "#613099"],
                    ],
                    colorbar=dict(
                        title=dict(
                            text="Time of<br>Day", font=dict(color="#d8d8d8")
                        ),
                        x=0.93,
                        xpad=0,
                        nticks=24,
                        tickfont=dict(color="#d8d8d8"),
                        thicknessmode="pixels",
                    ),
                ),
            ),
            # Plot important NYC landmarks as larger points
            Scattermapbox(
                lat=[list_of_locations[i]["lat"] for i in list_of_locations],
                lon=[list_of_locations[i]["lon"] for i in list_of_locations],
                mode="markers",
                hoverinfo="text",
                text=[i for i in list_of_locations],
                marker=dict(size=8, color="#ffa0a0"),
            ),
        ],
        layout=Layout(
            autosize=True,
            margin=go.layout.Margin(l=0, r=35, t=0, b=0),
            showlegend=False,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                center=dict(lat=latInitial, lon=lonInitial),  # Center map
                style="dark",
                bearing=bearing,
                zoom=zoom,
            ),
            updatemenus=[
                dict(
                    buttons=([
                        dict(
                            args=[
                                {
                                    "mapbox.zoom": 12,
                                    "mapbox.center.lon": "-73.991251",
                                    "mapbox.center.lat": "40.7272",
                                    "mapbox.bearing": 0,
                                    "mapbox.style": "dark",
                                }
                            ],
                            label="Reset Zoom",
                            method="relayout",
                        )
                    ]),
                    direction="left",
                    pad={"r": 0, "t": 0, "b": 0, "l": 0},
                    showactive=False,
                    type="buttons",
                    x=0.45,
                    y=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="#323130",
                    borderwidth=1,
                    bordercolor="#6d6d6d",
                    font=dict(color="#FFFFFF"),
                )
            ],
        ),
    )


# --- Main Entrypoint ---
if __name__ == "__main__":
    app.run(debug=True)
