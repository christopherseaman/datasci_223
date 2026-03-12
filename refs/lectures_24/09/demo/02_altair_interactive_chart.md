# Demo 2: Interactive Altair Chart - Simplified Wearable Health Insights

**Goal:** Create clear, functional, and engaging interactive Altair charts using a synthetic dataset simulating daily health metrics from wearable devices. This demo prioritizes robust chart rendering and understandable interactivity.

## Step 1: Set Up Your Environment

Ensure all required packages from `requirements.txt` are installed.

```python
# Import required packages
import altair as alt
import pandas as pd
import numpy as np
from IPython.display import HTML, Image, display # Ensure display is imported
import os # For checking file existence

# Enable Altair to render in notebooks
alt.data_transformers.enable('json')
```

## Step 2: Create Synthetic Wearable Device Data

We'll generate a dataset simulating daily health metrics over a month for several individuals. This version simplifies some data generation aspects to ensure clarity for visualization.

```python
# Generate synthetic wearable device data
np.random.seed(42)  # For reproducibility

n_users = 10 # Reduced user count for potentially clearer individual plots
n_days = 30
data = []

genders = ['Male', 'Female']
activity_profiles_config = {
    'Sedentary': {'base_steps': 3000, 'hr_factor': 1.03, 'stress_add': 1.5, 'sleep_hours_base': 6.8},
    'Lightly Active': {'base_steps': 6000, 'hr_factor': 1.0, 'stress_add': 0.5, 'sleep_hours_base': 7.2},
    'Active': {'base_steps': 10000, 'hr_factor': 0.97, 'stress_add': -1.0, 'sleep_hours_base': 7.7}
}
profile_keys = list(activity_profiles_config.keys())

for user_i in range(n_users):
    user_id = f"User{201+user_i}"
    age = np.random.randint(28, 58)
    gender = np.random.choice(genders)
    user_profile_key = np.random.choice(profile_keys)
    user_profile = activity_profiles_config[user_profile_key]

    for day_j in range(1, n_days + 1):
        # Daily variation
        steps_variation = np.random.normal(1, 0.25) # Multiplicative factor
        steps_taken = int(user_profile['base_steps'] * steps_variation)
        steps_taken = max(1000, min(22000, steps_taken))

        calories_burned = int(1600 + (steps_taken * 0.05) + (np.random.normal(0, 150)) + (100 if gender == 'Male' else 0) - (age*2))
        calories_burned = max(1300, calories_burned)

        avg_heart_rate = 70 - (age - 40)*0.25 + (10000 - steps_taken)*0.0015 / user_profile['hr_factor'] + np.random.normal(0,2.5)
        avg_heart_rate = int(max(50, min(100, avg_heart_rate))) # Daily average, not peak

        sleep_hours = user_profile['sleep_hours_base'] + np.random.normal(0, 0.6) - (steps_taken / 20000) # High activity might slightly reduce
        sleep_hours = round(max(4.5, min(9.5, sleep_hours)), 1)
        
        stress_level = 4 + user_profile['stress_add'] - (sleep_hours - 7)*0.8 - (steps_taken / 5000) + np.random.normal(0,1.2)
        stress_level = round(max(0.5, min(9.5, stress_level)), 1) # Keep within 0-10 effectively
        
        data.append({
            'user_id': user_id, 'day_of_study': day_j, 'age': age, 'gender': gender,
            'activity_profile': user_profile_key,
            'steps_taken': steps_taken, 'avg_heart_rate': avg_heart_rate,
            'sleep_hours': sleep_hours, 'stress_level': stress_level,
            'calories_burned': calories_burned
        })

wearable_df = pd.DataFrame(data)
print(f"Generated {len(wearable_df)} daily records for {wearable_df['user_id'].nunique()} users.")
print(wearable_df.head())
```

## Step 3: Basic Interactive Plot - Daily Steps Over Study Period

Visualizing daily steps for each user, colored by their activity profile. Scroll wheel zoom on the y-axis is disabled for easier navigation.

```python
steps_over_time_chart = alt.Chart(wearable_df).mark_line(point=alt.OverlayMarkDef(size=20)).encode(
    x=alt.X('day_of_study:Q', title='Day of Study', axis=alt.Axis(tickCount=n_days//3, grid=False)),
    y=alt.Y('steps_taken:Q', title='Daily Steps Taken', scale=alt.Scale(zero=False)),
    color=alt.Color('activity_profile:N', title='Activity Profile'),
    strokeDash=alt.StrokeDash('user_id:N', title='User ID', legend=None), 
    tooltip=[
        'user_id:N', 'day_of_study:Q', 'steps_taken:Q', 
        'activity_profile:N', 'avg_heart_rate:Q', 'stress_level:Q', 'sleep_hours:Q'
    ]
).properties(
    title='Daily Steps Over Study Period by User and Activity Profile',
    width=650, 
    height=300
).interactive(bind_y=False) # Disable y-axis scroll zoom, allow x-axis pan/zoom

steps_over_time_chart
```

## Step 4: Brush Selection - Heart Rate vs. Calories Burned

Exploring the relationship between average daily heart rate and calories burned. Scroll wheel zoom on the y-axis is disabled.

```python
hr_calories_brush_sel = alt.selection_interval(name='hr_calories_brush_selection')

hr_calories_scatter_chart = alt.Chart(wearable_df).mark_circle(size=70, opacity=0.8).encode(
    x=alt.X('avg_heart_rate:Q', title='Average Daily Heart Rate (bpm)', scale=alt.Scale(zero=False)),
    y=alt.Y('calories_burned:Q', title='Calories Burned', scale=alt.Scale(zero=False)),
    color=alt.condition(
        hr_calories_brush_sel,
        alt.Color('activity_profile:N', title='Activity Profile'),
        alt.value('lightgray')
    ),
    tooltip=['user_id:N', 'avg_heart_rate:Q', 'calories_burned:Q', 'steps_taken:Q', 'stress_level:Q']
).add_params(
    hr_calories_brush_sel
).properties(
    title='Heart Rate vs. Calories Burned (Brush to Select)',
    width=500,
    height=300
).interactive(bind_y=False) # Disable y-axis scroll zoom

hr_calories_scatter_chart
```

## Step 5: Interactive Filtering with Dropdown Selection

Create a simple dropdown filter to explore how different activity profiles affect sleep patterns.

```python
print("Creating interactive chart with dropdown filter...")

# Create a parameter for the dropdown selection
activity_filter = alt.param(
    name='activity_filter',
    bind=alt.binding_select(
        options=['All'] + sorted(wearable_df['activity_profile'].unique().tolist()),
        name='Activity Profile: '
    ),
    value='All'
)

# Create the chart with dropdown filtering
sleep_steps_chart = alt.Chart(wearable_df).add_params(
    activity_filter
).transform_filter(
    # Filter data based on dropdown selection
    alt.expr.if_(
        activity_filter == 'All',
        True,
        alt.datum.activity_profile == activity_filter
    )
).mark_circle(size=80, opacity=0.7).encode(
    x=alt.X('steps_taken:Q', title='Daily Steps', scale=alt.Scale(zero=False)),
    y=alt.Y('sleep_hours:Q', title='Sleep Hours', scale=alt.Scale(domain=[4, 10])),
    color=alt.Color('activity_profile:N', title='Activity Profile'),
    tooltip=['user_id:N', 'steps_taken:Q', 'sleep_hours:Q', 'stress_level:Q', 'activity_profile:N']
).properties(
    title='Sleep vs Steps by Activity Profile (Use dropdown to filter)',
    width=500,
    height=300
)

# Display the chart
sleep_steps_chart
```

## Step 6: Save Interactive Charts for Reports

Learn how to export Altair charts in different formats for embedding in reports and presentations.

```python
import os

# Create charts directory if it doesn't exist
charts_dir = 'charts'
os.makedirs(charts_dir, exist_ok=True)

# Save the interactive chart from Step 5 in multiple formats
chart_name = 'sleep_steps_interactive'

# 1. Save as JSON (preserves all interactivity)
json_path = os.path.join(charts_dir, f'{chart_name}.json')
sleep_steps_chart.save(json_path)
print(f"✅ Saved JSON: {json_path}")

# 2. Save as HTML (standalone interactive file)
html_path = os.path.join(charts_dir, f'{chart_name}.html')
sleep_steps_chart.save(html_path)
print(f"✅ Saved HTML: {html_path}")

# 3. Save as PNG (static image for presentations)
try:
    png_path = os.path.join(charts_dir, f'{chart_name}.png')
    sleep_steps_chart.save(png_path, scale_factor=2.0)
    print(f"✅ Saved PNG: {png_path}")
except Exception as e:
    print(f"⚠️  PNG save failed: {e}")
    print("   Install vl-convert-python for PNG export: pip install vl-convert-python")

print(f"\n📁 All charts saved to '{charts_dir}/' directory")
print("💡 Use HTML files for interactive web reports")
print("💡 Use PNG files for static presentations")
```

## Step 7: "Daily Wellness Dashboard" - Simplified

A less cluttered dashboard focusing on daily steps, sleep, and stress. Y-axis scroll zoom is disabled.

```python
# Simplified Wellness Dashboard
wellness_dashboard_v2 = alt.Chart(wearable_df).mark_circle(size=70, opacity=0.6, stroke='black', strokeWidth=0.2).encode(
    x=alt.X('steps_taken:Q', title='Daily Steps', scale=alt.Scale(zero=False)),
    y=alt.Y('sleep_hours:Q', title='Sleep (Hours)', scale=alt.Scale(domain=[4,10])),
    color=alt.Color('stress_level:Q', title='Stress Level', 
                  scale=alt.Scale(scheme='redyellowgreen', reverse=True)), # Lower stress = greener
    tooltip=[
        'user_id:N', 'day_of_study:Q', 'steps_taken:Q', 'sleep_hours:Q', 
        'stress_level:Q', 'activity_profile:N', 'avg_heart_rate:Q'
    ]
).properties(
    title={
        "text": "Daily Wellness Overview",
        "subtitle": "Steps vs. Sleep (Color = Stress Level)"
    },
    width=550, 
    height=350
).interactive(bind_y=False) # Y-axis scroll zoom already disabled by default on composite charts, but explicit for clarity

# Subtle reference lines
ideal_steps_ref = alt.Chart(pd.DataFrame({'ideal_steps': [8000]})).mark_rule(color='lightgray', strokeDash=[2,2]).encode(x='ideal_steps:Q')
ideal_sleep_ref = alt.Chart(pd.DataFrame({'ideal_sleep': [7.5]})).mark_rule(color='lightgray', strokeDash=[2,2]).encode(y='ideal_sleep:Q')

final_wellness_dashboard_v2 = (wellness_dashboard_v2 + ideal_steps_ref + ideal_sleep_ref).properties(
    background='#ffffff' # Plain background
)

json_path_wellness_v2 = os.path.join(charts_dir, 'wellness_dashboard_v2.json')
if 'final_wellness_dashboard_v2' in locals():
    try:
        final_wellness_dashboard_v2.save(json_path_wellness_v2)
        print(f"\nSaved {json_path_wellness_v2}")
    except Exception as e:
        print(f"⚠️ Error saving final_wellness_dashboard_v2: {e}")
else:
    print("⚠️ 'final_wellness_dashboard_v2' not defined. Skipping save.")

final_wellness_dashboard_v2
```

## Success Validation
This section remains the same, outlining the goals for the interactive charts.

- ✅ **Clear relationships** between variables in the synthetic data.
- ✅ **Engaging tooltips** providing rich contextual information.
- ✅ **Effective use of selections** (brush, dropdown) for data exploration.
- ✅ **Linked visualizations** where interactions in one chart update others.
- ✅ **Visually appealing** and informative "dashboard-style" compositions.
- ✅ **Saved outputs** (JSON, HTML, PNG) for various use cases.

## Bonus: Quick Data Quality Check

**Why perform data quality checks?**
Before diving into complex visualizations or analyses, it's crucial to perform basic data quality checks. This step helps us:
1.  **Identify potential errors:** Catch issues in data generation or loading (e.g., values outside expected ranges, incorrect data types).
2.  **Understand data distributions:** Get a feel for the typical values, ranges, and frequencies of different features in our dataset.
3.  **Ensure data makes sense:** Verify that the synthetic data, while not real, reflects plausible scenarios for the domain we're simulating (in this case, daily wearable metrics).
4.  **Prevent misleading visualizations:** Using flawed data can lead to incorrect interpretations and conclusions. Early checks mitigate this risk.

**What does the following code do?**
The Python code below will:
- Print the total number of records and unique users to confirm dataset size.
- Display the minimum and maximum values for key numerical features (age, steps, heart rate, sleep, stress, calories) to check their ranges.
- Show the distribution (value counts) for important categorical features (activity profile, gender) to understand their frequencies.
- Perform some example plausibility checks (e.g., ensuring stress levels are within 0-10, sleep hours are reasonable).
- Report any identified "issues" if values fall outside these predefined plausible ranges.

This provides a quick overview and helps build confidence in the data before we rely on it for visualization.

```python
# Quick sanity check on our new synthetic data
print("\nData Quality Check (Synthetic Wearable Data):")
if not wearable_df.empty:
    print(f"Total records: {len(wearable_df)}")
    print(f"Unique users: {wearable_df['user_id'].nunique()}")
    print(f"Age range: {wearable_df['age'].min()} - {wearable_df['age'].max()} years")
    print(f"Day of Study range: {wearable_df['day_of_study'].min()} - {wearable_df['day_of_study'].max()}")
    print(f"Steps Taken range: {wearable_df['steps_taken'].min()} - {wearable_df['steps_taken'].max()}")
    print(f"Avg Heart Rate range: {wearable_df['avg_heart_rate'].min()} - {wearable_df['avg_heart_rate'].max()} bpm")
    print(f"Sleep Hours range: {wearable_df['sleep_hours'].min():.1f} - {wearable_df['sleep_hours'].max():.1f} hrs")
    print(f"Stress Level range: {wearable_df['stress_level'].min():.1f} - {wearable_df['stress_level'].max():.1f}")
    print(f"Calories Burned range: {wearable_df['calories_burned'].min()} - {wearable_df['calories_burned'].max()}")
    
    print(f"\nPrimary Activity Profiles: {wearable_df['activity_profile'].value_counts().to_dict()}")
    print(f"Gender Distribution: {wearable_df['gender'].value_counts().to_dict()}")

    issues = []
    if not (0 <= wearable_df['stress_level'].min() and wearable_df['stress_level'].max() <= 10):
        issues.append(f"Stress level out of 0-10 bounds: min={wearable_df['stress_level'].min()}, max={wearable_df['stress_level'].max()}")
    if not (3 <= wearable_df['sleep_hours'].min() and wearable_df['sleep_hours'].max() <= 11): 
        issues.append(f"Sleep hours out of typical range: min={wearable_df['sleep_hours'].min()}, max={wearable_df['sleep_hours'].max()}")
    if wearable_df['steps_taken'].min() < 0:
        issues.append(f"Steps taken has negative values: {wearable_df['steps_taken'].min()}")
    
    if issues:
        print(f"\n⚠️  Potential data issues found: {issues}")
    else:
        print("\n✅ Basic data checks for plausible ranges passed.")
else:
    print("⚠️ wearable_df is empty, skipping data quality checks.")
```

This revised demo aims to provide a more compelling and educational example for students learning Altair with health-related data. 🏃‍♀️💤🧘‍♂️📊