# Demo 2: Physiological Stress Classification Using Wearable Sensor Data 🌡️

## Learning Objectives 🎯

By the end of this demo, you will be able to:
1. Load and preprocess multi-sensor physiological data
2. Extract meaningful features from time series signals
3. Build and evaluate tree-based classification models
4. Interpret model decisions using SHAP values

## Setup and Imports 🛠️

First, let's install required packages:

```python
%pip install -r requirements.txt --quiet
```

Now import the libraries we'll need:

```python
# Data manipulation
import pandas as pd
import numpy as np
from pathlib import Path

# Signal processing
from scipy import stats, signal

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap  # For model interpretation
```

## 1. Data Loading and Initial Exploration 📊

### Dataset Overview

This dataset contains physiological signals from students during exams, collected using wearable sensors. The signals include:

| Signal Type | Description | Unit | Sampling Rate |
|------------|-------------|------|---------------|
| Heart Rate (HR) | Beats per minute | BPM | 1 Hz |
| Electrodermal Activity (EDA) | Skin conductance | μS | 4 Hz |
| Blood Volume Pulse (BVP) | Blood volume changes | - | 64 Hz |
| Temperature | Skin temperature | °C | 4 Hz |
| Inter-Beat Interval (IBI) | Time between heartbeats | s | Variable |
| Accelerometer | 3-axis movement | g | 32 Hz |

### Time Series Visualization Reference Card 📈

When working with physiological time series data, consider these visualization tips:

| Plot Type | Best For | Key Parameters |
|-----------|----------|----------------|
| Line Plot | Continuous signals (HR, EDA) | `alpha` for overlapping lines |
| Scatter Plot | Sparse data (IBI) | `s` for point size |
| Multi-line | Multiple channels (ACC) | `label` for legend |
| Fill Between | Showing variance | `alpha` for transparency |

Common gotchas:
- Always check timestamps are in correct format
- Consider downsampling if plot is too dense
- Use appropriate y-axis scales for each signal
- Add clear labels and units

Let's first download the data if it's not already present:

```python
# Create data directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# Define the dataset directory name (using the actual name from the downloaded data)
dataset_dir = data_dir / 'a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0'

# Check if data exists and is in the correct structure
if (dataset_dir / 'Data').exists() and (dataset_dir / 'StudentGrades.txt').exists():
    print("Dataset already downloaded and in correct structure.")
    # Set data paths for later use
    data_path = dataset_dir / 'Data'
    grades_path = dataset_dir / 'StudentGrades.txt'
else:
    print("Downloading dataset from PhysioNet...")
    !wget -r -N -c -np https://physionet.org/files/wearable-exam-stress/1.0.0/ -P {data_dir}
    # Move files up from nested directory
    !mv {data_dir}/physionet.org/files/wearable-exam-stress/1.0.0/* {data_dir}/
    !rm -r {data_dir}/physionet.org
    print("Download complete!")
    # Set data paths after download
    data_path = dataset_dir / 'Data'
    grades_path = dataset_dir / 'StudentGrades.txt'
```

### Examining Student Performance

Let's look at the grade distribution to understand our target variable:

```python
# Load and parse grades
print("\nLoading student grades...")

def parse_grades(file_path):
    with open(file_path, 'r', encoding='latin1') as f:
        lines = f.readlines()
    
    # Initialize lists for each exam
    midterm1 = []
    midterm2 = []
    final = []
    
    current_exam = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if 'MIDTERM 1' in line:
            current_exam = midterm1
        elif 'MIDTERM 2' in line:
            current_exam = midterm2
        elif 'FINAL' in line:
            current_exam = final
        elif line.startswith('S'):  # Student grade line
            # Find the grade after "SXX" (where XX is the student number)
            try:
                # The grade is everything after the student ID (S01, S02, etc.)
                parts = line.split('S')[1].split()  # Split on whitespace
                if len(parts) >= 2:  # Should have at least student number and grade
                    grade = float(parts[-1])  # Take the last part as the grade
                    if current_exam is not None:
                        current_exam.append(grade)
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse line: {line}")
                continue
    
    # Create DataFrame
    grades_df = pd.DataFrame({
        'Midterm 1': midterm1,
        'Midterm 2': midterm2,
        'Final': [x/2 for x in final]  # Convert final to percentage
    })
    
    return grades_df

# Load and parse grades
grades_df = parse_grades(grades_path)
print("\nGrade Distribution Summary:")
print(grades_df.describe())

# Create an informative grade distribution plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=grades_df.melt(), x='variable', y='value', palette='viridis')
plt.title('Grade Distribution Across Exams')
plt.xlabel('Exam')
plt.ylabel('Grade (%)')

# Add individual points to show the actual distribution
sns.swarmplot(data=grades_df.melt(), x='variable', y='value', 
              color='red', alpha=0.5, size=4)

plt.show()

print("\n📊 Key Observations:")
print(f"- Average grade across all exams: {grades_df.values.mean():.1f}%")
print(f"- Highest grade: {grades_df.values.max():.1f}%")
print(f"- Lowest grade: {grades_df.values.min():.1f}%")
print(f"- Number of students: {len(grades_df)}")
```

### Loading and Visualizing Single Subject Data

Let's create a function to load data for a single subject. This is a common pattern in physiological data analysis - start with one subject to develop your pipeline:

```python
def load_subject_data(subject_id, session, data_dir='data'):
    """Load all sensor data for a given subject and session.
    
    Args:
        subject_id (int): Subject ID (1-10)
        session (str): 'Midterm 1', 'Midterm 2', or 'Final'
        data_dir (str): Path to data directory
        
    Returns:
        dict: Dictionary of DataFrames for each sensor type
    """
    base_path = Path(data_dir) / f'S{subject_id}' / session
    data = {}
    
    # Load each sensor file
    for sensor in ['ACC', 'BVP', 'EDA', 'HR', 'IBI', 'TEMP']:
        try:
            df = pd.read_csv(base_path / f'{sensor}.csv')
            # Add timestamp column - crucial for time series analysis
            if sensor != 'IBI':  # IBI has its own timestamp format
                df['timestamp'] = pd.to_datetime(df.iloc[:, 0], unit='s')
            data[sensor] = df
        except FileNotFoundError:
            print(f'Warning: {sensor} data not found for subject {subject_id}, {session}')
    
    return data

# Load example data for Subject 1's first midterm
print("Loading data for Subject 1, Midterm 1...")
subject_data = load_subject_data(1, 'Midterm 1')

# Create a comprehensive visualization of all signals
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Physiological Signals During Exam - Subject 1, Midterm 1')

# Define signals to plot with their proper titles and units
signals = [
    ('HR', 'Heart Rate (bpm)'),
    ('EDA', 'Electrodermal Activity (μS)'),
    ('TEMP', 'Temperature (°C)'),
    ('BVP', 'Blood Volume Pulse'),
    ('ACC', 'Acceleration (g)'),
    ('IBI', 'Inter-Beat Interval (s)')
]

# Plot each signal with appropriate visualization type
for (signal_type, title), ax in zip(signals, axes.flat):
    if signal_type in subject_data:
        df = subject_data[signal_type]
        if signal_type == 'ACC':
            # Accelerometer has 3 axes - use multi-line plot
            ax.plot(df['timestamp'], df.iloc[:, 1], 'r-', label='X', alpha=0.5)
            ax.plot(df['timestamp'], df.iloc[:, 2], 'g-', label='Y', alpha=0.5)
            ax.plot(df['timestamp'], df.iloc[:, 3], 'b-', label='Z', alpha=0.5)
            ax.legend()
        elif signal_type == 'IBI':
            # IBI data is sparse - use scatter plot
            ax.scatter(pd.to_datetime(df.iloc[:, 0], unit='s'), df.iloc[:, 1], 
                      alpha=0.5, s=10)
        else:
            # Regular line plot for continuous signals
            ax.plot(df['timestamp'], df.iloc[:, 1], '-', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

# 🧠 Comprehension Check:
# 1. Why do we use different plot types for different signals?
# 2. What patterns might indicate stress in these signals?
# 3. Why is the IBI data more sparse than other signals?

## 2. Feature Extraction for Stress Detection 🔍

### Time Series Feature Engineering Reference Card 📊

When working with physiological time series data, we typically extract these types of features:

| Feature Type | Description | Examples | Use Case |
|--------------|-------------|----------|-----------|
| Statistical | Basic statistics of signal | mean, std, min, max | General signal characteristics |
| Temporal | Time-based patterns | peaks, intervals, rate of change | Event detection |
| Frequency | Periodic patterns | dominant frequency, power spectrum | Rhythmic patterns |
| Derived | Combinations of signals | activity level from ACC | Complex behaviors |

Common approaches:
1. **Sliding Windows**: Analyze signal in fixed-time chunks
2. **Peak Detection**: Find significant signal changes
3. **Signal Smoothing**: Remove noise before feature extraction
4. **Cross-Signal Features**: Combine multiple sensors

Now we'll extract features that might indicate stress levels using a sliding window approach:

```python
def extract_window_features(data, window_size=60, overlap=30):
    """Extract features from a sliding window of physiological data.
    
    Args:
        data (dict): Dictionary of sensor DataFrames
        window_size (int): Window size in seconds
        overlap (int): Window overlap in seconds
        
    Returns:
        pd.DataFrame: Features for each window
    """
    features = []
    
    # Get start and end times across all sensors
    start_time = max(df['timestamp'].min() for df in data.values() if 'timestamp' in df.columns)
    end_time = min(df['timestamp'].max() for df in data.values() if 'timestamp' in df.columns)
    
    # Create sliding windows with overlap
    current_time = start_time
    while current_time + pd.Timedelta(seconds=window_size) <= end_time:
        window_end = current_time + pd.Timedelta(seconds=window_size)
        window_features = {}
        
        # Process each signal type
        for signal_type, df in data.items():
            if signal_type == 'ACC':
                # Activity level from 3-axis acceleration
                window_data = df[(df['timestamp'] >= current_time) & 
                               (df['timestamp'] < window_end)]
                if len(window_data) > 0:
                    # Calculate magnitude of acceleration vector
                    acc_mag = np.sqrt(window_data.iloc[:, 1]**2 + 
                                    window_data.iloc[:, 2]**2 + 
                                    window_data.iloc[:, 3]**2)
                    window_features.update({
                        'activity_mean': acc_mag.mean(),  # Average activity level
                        'activity_std': acc_mag.std(),    # Activity variability
                        'activity_max': acc_mag.max()     # Peak activity
                    })
            
            elif signal_type == 'HR':
                # Heart rate features
                window_data = df[(df['timestamp'] >= current_time) & 
                               (df['timestamp'] < window_end)]
                if len(window_data) > 0:
                    hr = window_data.iloc[:, 1]
                    window_features.update({
                        'hr_mean': hr.mean(),     # Average heart rate
                        'hr_std': hr.std(),       # Heart rate variability
                        'hr_max': hr.max(),       # Maximum heart rate
                        'hr_min': hr.min()        # Minimum heart rate
                    })
            
            elif signal_type == 'EDA':
                # Electrodermal activity features
                window_data = df[(df['timestamp'] >= current_time) & 
                               (df['timestamp'] < window_end)]
                if len(window_data) > 0:
                    eda = window_data.iloc[:, 1]
                    # Find peaks (potential stress responses)
                    peaks, _ = signal.find_peaks(eda, height=eda.mean(), 
                                               distance=int(len(eda)/10))
                    window_features.update({
                        'eda_mean': eda.mean(),     # Tonic (baseline) component
                        'eda_std': eda.std(),       # EDA variability
                        'eda_peaks': len(peaks),    # Number of SCRs (skin conductance responses)
                        'eda_max': eda.max()        # Maximum conductance
                    })
        
        if window_features:
            window_features['window_start'] = current_time
            window_features['window_end'] = window_end
            features.append(window_features)
        
        # Slide window forward by overlap amount
        current_time += pd.Timedelta(seconds=overlap)
    
    return pd.DataFrame(features)

# Extract features for our example subject
print("Extracting features from physiological signals...")
features_df = extract_window_features(subject_data)
print(f"Extracted {len(features_df.columns)-2} features across {len(features_df)} windows")

# Visualize key features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Key Features Over Time - Subject 1, Midterm 1')

# Plot heart rate statistics with confidence band
axes[0, 0].plot(features_df['window_start'], features_df['hr_mean'], 'b-', 
                label='Mean HR', linewidth=2)
axes[0, 0].fill_between(features_df['window_start'], 
                       features_df['hr_mean'] - features_df['hr_std'],
                       features_df['hr_mean'] + features_df['hr_std'], 
                       alpha=0.2, color='blue', label='±1 SD')
axes[0, 0].set_title('Heart Rate Variability')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('BPM')
axes[0, 0].legend()

# Plot EDA features
axes[0, 1].plot(features_df['window_start'], features_df['eda_mean'], 'r-', 
                label='Mean EDA', linewidth=2)
axes[0, 1].plot(features_df['window_start'], features_df['eda_peaks'], 'g-', 
                label='EDA Peaks', linewidth=2)
axes[0, 1].set_title('Electrodermal Activity')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('μS / Peak Count')
axes[0, 1].legend()

# Plot activity level with confidence band
axes[1, 0].plot(features_df['window_start'], features_df['activity_mean'], 'g-',
                label='Mean Activity', linewidth=2)
axes[1, 0].fill_between(features_df['window_start'],
                       features_df['activity_mean'] - features_df['activity_std'],
                       features_df['activity_mean'] + features_df['activity_std'],
                       alpha=0.2, color='green', label='±1 SD')
axes[1, 0].set_title('Physical Activity Level')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Acceleration (g)')
axes[1, 0].legend()

# Plot feature correlations
corr = features_df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, ax=axes[1, 1], cmap='RdBu_r', center=0, 
            xticklabels=True, yticklabels=True)
axes[1, 1].set_title('Feature Correlations')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# 🧠 Comprehension Check:
# 1. Why do we use overlapping windows instead of non-overlapping ones?
# 2. What might a high number of EDA peaks indicate?
# 3. Looking at the correlation heatmap, which features seem most related?

print("\n📊 Feature Summary:")
print("Statistical Features:")
for col in features_df.columns:
    if col not in ['window_start', 'window_end']:
        print(f"- {col}:")
        print(f"  Mean: {features_df[col].mean():.2f}")
        print(f"  Std:  {features_df[col].std():.2f}")
```

### Understanding the Features

Let's examine what each feature tells us about potential stress:

1. **Heart Rate Features**
   - `hr_mean`: Average heart rate - elevated during stress
   - `hr_std`: Heart rate variability - often decreases under stress
   - `hr_max/min`: Extreme values might indicate stress responses

2. **EDA Features**
   - `eda_mean`: Baseline skin conductance - increases with stress
   - `eda_peaks`: Skin conductance responses - more frequent during stress
   - `eda_std`: Variability in skin response
   - `eda_max`: Maximum conductance - peaks during stress

3. **Activity Features**
   - `activity_mean`: Overall movement level
   - `activity_std`: Movement variability
   - `activity_max`: Sudden movements/gestures

These features will form the basis for our stress detection model in the next section.

## 0. Project Overview

### Research Questions

1. Can we identify periods of high stress in students using physiological signals during exam sessions?
2. Do stress response patterns correlate with academic performance?

### Data Location

`wget -r -N -c -np https://physionet.org/files/wearable-exam-stress/1.0.0/`

### Classification Objectives

**Two-Stage Classification Task:**

1. **Stress Level Detection:**
   - Identify high-stress periods from physiological signals
   - Use sliding windows to detect stress state changes

2. **Performance Analysis:**
   - Group students by exam performance (above/below median)
   - Analyze stress response patterns between groups

### Analysis Thresholds

- **Stress Level:** Defined by significant deviations in physiological signals
- **Performance Groups:**
    - High: Students scoring above the median grade
    - Low: Students scoring below the median grade

## 1. Dataset Context

### Data Source

- Wearable Exam Stress Dataset from PhysioNet
- Empatica E4 wristband recordings
- 10 participants across three exam sessions
- All exams start at 9:00 AM (CT/CDT)
- Midterms: 1.5 hours, Finals: 3 hours

### Classification Features

1. Physiological Signals
   - Heart Rate
   - Electrodermal Activity (EDA)
   - Blood Volume Pulse
   - Skin Temperature
   - Inter-Beat Interval (IBI)
   - 3-axis Accelerometer

2. Derived Features
   - Heart Rate Variability (HRV) metrics
   - Frequency domain characteristics
   - Time-window statistical features
   - Movement/activity levels

## 2. Preprocessing Strategy

### Data Cleaning

- Handle missing values (<1%)
- Remove physiological outliers using rolling z-scores
- Normalize signal intensities
- Synchronize timestamps across sensors
- Filter motion artifacts using accelerometer data

### Feature Extraction Approach

1. **Time-Window Features (Stress Detection)**
   - Sliding windows (60-second, 30-second overlap)
   - Statistical features: mean, std, min, max
   - Physiological features: HRV, EDA peaks
   - Activity level from accelerometer

2. **Session-Level Features (Performance Analysis)**
   - Aggregate stress measures per exam
   - Stress pattern characteristics
   - Temporal stress distribution

## 3. Building and Evaluating Classification Models 🤖

### Model Selection Reference Card 📋

When choosing a model for physiological data classification:

| Model Type | Strengths | Considerations | Best For |
|------------|-----------|----------------|-----------|
| Random Forest | Handles nonlinear patterns, feature importance | Can be slow with many trees | General-purpose, robust |
| XGBoost | Fast, often best performance | More hyperparameters to tune | When performance is key |
| Logistic Regression | Simple, interpretable | Only linear patterns | Baseline model |
| Neural Networks | Complex patterns, time series | Needs lots of data | Deep patterns |

Let's implement and compare Random Forest and XGBoost for stress detection:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# First, let's create our target variable (stress vs. no stress)
# We'll use a simple threshold-based approach for this demo
def create_stress_labels(features_df, hr_threshold=75, eda_threshold=1.5):
    """Create binary stress labels based on physiological thresholds."""
    return ((features_df['hr_mean'] > hr_threshold) & 
            (features_df['eda_peaks'] > eda_threshold)).astype(int)

# Create labels and prepare features
y = create_stress_labels(features_df)
feature_cols = [col for col in features_df.columns 
                if col not in ['window_start', 'window_end']]
X = features_df[feature_cols]

# Split data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dataset shapes:")
print(f"Training: {X_train.shape}, {y_train.shape}")
print(f"Testing:  {X_test.shape}, {y_test.shape}")
print(f"\nClass distribution:")
print(f"Stress periods: {y.mean()*100:.1f}%")
print(f"Non-stress periods: {(1-y.mean())*100:.1f}%")

# Train and evaluate Random Forest
print("\n🌳 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_pred))

# Train and evaluate XGBoost
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

print("\nXGBoost Results:")
print(classification_report(y_test, xgb_pred))

# Visualize feature importance for both models
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Feature Importance Comparison')

# Random Forest importance
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

sns.barh(data=rf_importance, y='feature', x='importance', ax=axes[0])
axes[0].set_title('Random Forest')
axes[0].set_xlabel('Importance')

# XGBoost importance
xgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=True)

sns.barh(data=xgb_importance, y='feature', x='importance', ax=axes[1])
axes[1].set_title('XGBoost')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()

# 🧠 Comprehension Check:
# 1. Which model performed better? Why might that be?
# 2. What are the most important features for stress detection?
# 3. How might we improve the model's performance?

### Model Interpretation with SHAP Values 🔍

SHAP (SHapley Additive exPlanations) values help us understand how each feature contributes to individual predictions:

```python
# Calculate SHAP values for XGBoost model
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

# Plot SHAP summary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_cols)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.show()

# Plot SHAP dependence for top features
top_features = xgb_importance.tail(3)['feature'].values
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('SHAP Dependence Plots for Top Features')

for i, feature in enumerate(top_features):
    shap.dependence_plot(feature, shap_values, X_test_scaled, 
                        feature_names=feature_cols, ax=axes[i])
    axes[i].set_title(feature)

plt.tight_layout()
plt.show()

print("\n🎯 Key Findings:")
print("1. Most important stress indicators:")
for feature, importance in xgb_importance.tail(3).iloc[::-1].values:
    print(f"   - {feature}: {importance:.3f}")

print("\n2. Model Performance:")
print(f"   - Random Forest Accuracy: {rf_model.score(X_test_scaled, y_test):.3f}")
print(f"   - XGBoost Accuracy: {xgb_model.score(X_test_scaled, y_test):.3f}")

print("\n3. Potential Improvements:")
print("   - Collect more training data")
print("   - Engineer additional features (e.g., frequency domain)")
print("   - Try different window sizes")
print("   - Incorporate temporal dependencies")
```

### Understanding SHAP Values 📊

SHAP values tell us how much each feature contributed to a prediction:
- **Red** points indicate high feature values
- **Blue** points indicate low feature values
- **Position** shows whether the feature pushed the prediction higher (right) or lower (left)
- **Spread** shows the range of impact for each feature

Key insights from our model:
1. EDA peaks are strong indicators of stress
2. Heart rate variability (hr_std) is inversely related to stress
3. Physical activity can confound stress detection

## Learning Objectives

1. Process and analyze multivariate physiological time series
2. Implement two-stage classification pipeline
3. Extract meaningful features from wearable sensor data
4. Apply cross-validation for time series data
5. Interpret stress patterns and their relationship to performance

## Comprehension Checkpoint

1. How can we identify stress periods from physiological signals?
2. What challenges exist in personalizing stress detection?
3. Why use a two-stage approach for this analysis?
4. How might different stress responses relate to performance?

## Recommended Workflow

1. Data exploration and cleaning
2. Signal preprocessing and feature extraction
3. Stress detection model development
4. Performance analysis model development
5. Model interpretation and visualization
6. Results analysis and validation

## 4. Conclusions and Next Steps 🎓

### What We've Learned 📚

In this demo, we've covered:
1. Loading and visualizing multi-sensor physiological data
2. Extracting meaningful features from time series signals
3. Building and comparing tree-based classification models
4. Interpreting model decisions using SHAP values

Key takeaways:
- Physiological signals contain rich information about stress levels
- Feature engineering is crucial for time series classification
- Tree-based models work well for this type of data
- Model interpretation helps validate our approach

### Challenges and Limitations 🤔

1. **Data Quality**
   - Missing data periods
   - Sensor noise and artifacts
   - Individual differences in physiological responses

2. **Methodological**
   - Simple threshold-based labeling
   - Limited to single-subject analysis
   - No validation against self-reported stress

3. **Technical**
   - Basic feature set
   - No temporal dependencies considered
   - Limited cross-validation

### Future Directions 🚀

1. **Data Collection**
   - Collect self-reported stress levels
   - Include more subjects and conditions
   - Add environmental context

2. **Feature Engineering**
   - Add frequency domain features
   - Include cross-signal features
   - Try deep learning for automatic feature extraction

3. **Modeling**
   - Implement temporal models (LSTM, GRU)
   - Use multi-task learning
   - Explore transfer learning

### Additional Resources 📚

1. Papers:
   - [Wearable Sensors for Monitoring Stress](https://www.mdpi.com/1424-8220/18/8/2619)
   - [Deep Learning for Physiological Signal Analysis](https://www.nature.com/articles/s41551-020-0538-5)

2. Tools:
   - [BioSPPy](https://github.com/PIA-Group/BioSPPy) - Biosignal Processing in Python
   - [NeuroKit2](https://github.com/neuropsychology/NeuroKit) - Neurophysiological Signal Processing

3. Datasets:
   - [WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
   - [AffectiveROAD](https://www.nature.com/articles/sdata201743)

# 🧠 Final Comprehension Check

1. What are the key steps in processing physiological sensor data for stress detection?
2. How do different physiological signals relate to stress levels?
3. What are the main advantages and limitations of our approach?
4. How would you improve this system for real-world deployment?

Remember: This is just a starting point! Real-world stress detection systems need careful validation, ethical considerations, and robust error handling. Keep exploring and experimenting! 🌟