> [!important]  
> Note: Always use UTC or epoch (“unix time”) for timestamps to avoid complications with time zones, daylight savings, and leap years/seconds  

# **Fundamentals**

Time series data is characterized by sequential measurements over intervals. Understanding its components—trend, seasonality, and noise—is crucial for effective analysis.

## **Key Concepts:**

- **Trend**: The underlying pattern in the data over time
- **Seasonality**: Regular variations tied to time intervals, such as days, months, or seasons
- **Noise**: Random fluctuations that obscure the underlying pattern

## **Challenges in Time Series Analysis**

- **Handling Missing Values**: Time series data often comes with gaps that need to be addressed carefully.
- **Stationarity**: Many time series methods assume the data is stationary. Transformations or differencing may be required
- **Choosing the Right Model**: No one-size-fits-all. Model selection depends on the specific characteristics of the data
- **Model Complexity and Overfitting**: As models become more sophisticated to capture complex patterns, there's a risk of overfitting to historical data, making it less generalizable to future data

## Applications

- **Patient Monitoring:** Analyzing time series data from patient monitoring devices can help in early detection of deteriorating health conditions
- **Resource Allocation**: Time series analysis can predict hospital admissions or disease outbreaks, aiding in efficient resource allocation and preparedness
- **Disease Trend Analysis:** Studying the incidence and prevalence of diseases over time can inform public health strategies and interventions
- **Treatment Effectiveness Over Time:** Longitudinal data can shed light on how patients respond to treatments across different time frames

# ML/Stats Methods

## Summary

- **Cross-Validation for Time-Series (TimeSeriesSplit)**: For cross-validation in time series, respecting the temporal order of data.
- **ARIMA**: Suitable for forecasting when data shows trends or seasonality, requiring parameter tuning (p, d, q).
- **Cox Regression**: Used in survival analysis, modeling the time until an event occurs, factoring in various covariates.
- **Kaplan-Meier**: Analyzes duration until events, useful in diverse fields like medicine and engineering.
- **Advanced Topics:**
    - **Vector Autoregression**: Captures interdependencies in multivariate time series data.
    - **Seasonal Decomposition**: Breaks down time series into trend, seasonal, and residual components.
    - **Dynamic Time Warping**: Measures similarity between two temporal sequences, useful for varying speeds.
    - **State Space Models and Kalman Filters**: For modeling observed and unobserved variables in time series, with Kalman Filters estimating hidden states.
    - **Generalized Additive Models**: Flexible approach for non-linear time series data relationships.
    - **Prophet**: Forecasts time series data with trends and seasonality, fitting yearly, weekly, and daily patterns.

## **Cross-Validation for Time Series**

Unlike traditional cross-validation, time series cross-validation involves rolling or expanding windows due to the ordered nature of the data.

![[Notion/Getting into Data Science/Applied Data Science with Python/It’s About Time/Untitled.png|Untitled.png]]

```Python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit()
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

## ARIMA (AutoRegressive Integrated Moving Average)

![[Notion/Getting into Data Science/Applied Data Science with Python/It’s About Time/Untitled 1.png|Untitled 1.png]]

ARIMA is a popular statistical method for time series forecasting that models the data as a linear combination of its past values (autoregressive part), differences of past values (integrated part), and past forecast errors (moving average part). It's effective for data with trends and/or seasonality.

```Python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Load your time series data
# data = pd.read_csv('your_time_series.csv')

# Initialize and fit the ARIMA model
# The order (p,d,q) needs to be determined based on your data
model = ARIMA(data, order=(p, d, q))
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=5)
```

### **Parameters**

- `p`: Autoregressive order, number of lag observations included in the model, also known as the lag order
- `d`: Differencing order; number of times the raw observations are differenced, also referred to as the degree of differencing
- `q`: Moving average order, specifies the size of the moving average window, or the order of the moving average component of the model

## Cox Regression

![[Notion/Getting into Data Science/Applied Data Science with Python/It’s About Time/Untitled 2.png|Untitled 2.png]]

Cox Regression, also known as the Cox Proportional Hazards Model, is used in survival analysis to model the time until an event occurs, considering the impact of various covariates. It's a semi-parametric model that estimates the hazard (or risk) of the event occurring at a certain time.

```Python
from lifelines import CoxPHFitter

# Assuming `df` is your DataFrame with 'duration' and 'event' columns
# and additional columns for covariates
# coxph = CoxPHFitter()
# coxph.fit(df, duration_col='duration', event_col='event')

# Predicting the hazard
# predictions = coxph.predict_survival_function(df)
```

### **Parameters**

- `duration_col`: Column in DataFrame that contains the duration until the event or censoring
- `event_col`: Column in DataFrame that indicates if the event of interest occurred

## Survival Analysis

![[Notion/Getting into Data Science/Applied Data Science with Python/It’s About Time/Untitled 3.png|Untitled 3.png]]

Kaplan-Meier survival analysis involves statistical methods for analyzing the expected duration until one or more events happen, like death in biological organisms and failure in mechanical systems. It's used in medicine, biology, engineering, economics, and many other fields.

```Python
from lifelines import KaplanMeierFitter

# Assuming `df` is your DataFrame with 'duration' and 'event' columns
kmf = KaplanMeierFitter()
kmf.fit(durations=df["duration"], event_observed=df["event"])

# Plotting the survival function
kmf.plot_survival_function()
```

### **Parameters**

- `durations`: Duration until an event or censoring
- `event_observed`: Whether the event of interest occurred

## Advanced Topics

### **Vector Autoregression**

A multivariate statistical model used to capture the linear interdependencies among multiple time series. Used for multivariate time series.

> [!info] Developing Vector AutoRegressive Model in Python!  
> Vector AutoRegressive (VAR) is a multivariate forecasting algorithm that is used when two or more time series influence each other.  
> [https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/](https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/)  

```Python
from statsmodels.tsa.api import VAR

model = VAR(endog=df)
model_fit = model.fit(maxlags=15, ic='aic')
```

### **Seasonal Decomposition**

![[Notion/Getting into Data Science/Applied Data Science with Python/It’s About Time/Untitled 4.png|Untitled 4.png]]

Decomposing a time series means breaking it down into its constituent components: trend, seasonal, and residual.

```Python
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform decomposition
result = seasonal_decompose(data, model='additive', period=12)

# Plot the decomposed components
result.plot()
```

### **Dynamic Time Warping**

![[IMG_0095.webp]]

An algorithm for measuring similarity between two temporal sequences which may vary in speed.

```Python
from dtaidistance import dtw

distance = dtw.distance(sequence1, sequence2)
```

### **State Space Models and Kalman Filters**

A framework for modeling time series data that allows incorporating both observed and unobserved variables, with Kalman Filters providing a way to infer hidden states.

```Python
from pykalman import KalmanFilter

kf = KalmanFilter(initial_state_mean=0, n_dim_obs=2)
(filtered_state_means, filtered_state_covariances) = kf.filter(data)
```

### **Generalized Additive Models for Time Series**

Flexible models that allow the data to determine the shape of the relationship between variables, useful for non-linear time series data.

```Python
from pygam import LinearGAM, s

gam = LinearGAM(s(0) + s(1)).fit(X, y)
gam.predict(X_new)
```

### **Prophet by Facebook**

![[Notion/Getting into Data Science/Applied Data Science with Python/It’s About Time/Untitled 5.png|Untitled 5.png]]

Prophet is a tool for forecasting time series data based on an additive model where non-linear trends fit with yearly, weekly, and daily seasonality.

- [**Time Series Forecasting with TensorFlow, ARIMA, and PROPHET**](https://polzinben.github.io/Time-Series-Forecasting/)

```Python
from fbprophet import Prophet

# Initialize the Prophet model
model = Prophet()

# Fit the model
model.fit(df)

# Make a future dataframe and forecast
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
```

# Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are a class of neural networks designed to recognize patterns in sequences of data, such as time series, genomes, handwriting, or spoken words. Unlike traditional neural networks, RNNs use their internal state (memory) to process variable length sequences of inputs.

![[Notion/Getting into Data Science/Applied Data Science with Python/It’s About Time/Untitled 6.png|Untitled 6.png]]

```Python
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Define the model
model = Sequential()
model.add(SimpleRNN(units=50, activation='tanh', input_shape=(None, 1)))
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
```

### Examples

- **Forecasting:** e.g., weather
- **Text Generation:** generating text character by character or word by word
- **Speech Recognition:** transcribing spoken words into text

## **Long Short-Term Memory (LSTM) Networks**

LSTMs are a type of RNN designed to remember information for long periods, making them ideal for time series data.

> [!info] The Complete LSTM Tutorial With Implementation  
> LSTMs are a stack of neural networks composed of linear layers; weights and biases.  
> [https://www.analyticsvidhya.com/blog/2022/01/the-complete-lstm-tutorial-with-implementation/](https://www.analyticsvidhya.com/blog/2022/01/the-complete-lstm-tutorial-with-implementation/)  

```Python
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define the model
model = Sequential()
model.add(LSTM(units=50, activation='tanh', input_shape=(time_steps, features)))
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
```

# LLM’s (Large Language Models)

> [!info] Paper page - Chronos: Learning the Language of Time Series  
> Join the discussion on this paper page  
> [https://huggingface.co/papers/2403.07815](https://huggingface.co/papers/2403.07815)  

## Example: Patient Trajectory Encoding

In the context of healthcare, LLMs can be adapted to encode patient trajectories instead of words. This involves training the model on patient data to predict future health events or outcomes based on past medical history, similar to how a language model predicts the next word in a sentence.

- [**Unlearn.ai**](https://www.unlearn.ai/)**:** A platform that creates "digital twins" of patients, which are computational models predicting individual patient trajectories.

This approach allows for personalized medicine by simulating how different treatments might affect a patient over time. The founder of [Unlearn.ai](http://unlearn.ai/) will be speaking in a seminar early May, potentially providing more insights into the application of LLMs in healthcare.

> [!info]  
>  
> [https://www.nature.com/articles/s41746-023-00879-8](https://www.nature.com/articles/s41746-023-00879-8)  

> [!info] A Transformer-Based Model for Zero-Shot Health Trajectory Prediction  
> Integrating modern machine learning and clinical decision-making has great promise for mitigating healthcare’s increasing cost and complexity.  
> [https://www.medrxiv.org/content/10.1101/2024.02.29.24303512v1.full](https://www.medrxiv.org/content/10.1101/2024.02.29.24303512v1.full)  

> [!info]  
>  
> [https://www.nature.com/articles/s41746-022-00742-2](https://www.nature.com/articles/s41746-022-00742-2)  

# Practice: Smartwatch Gestures

The "smartwatch_gestures" dataset consists of sensor data collected from smartwatches, capturing various gestures.

We can perform exploratory data analysis (EDA) to understand the characteristics of these gestures, including the distribution of gesture classes, basic statistics of sensor readings, and visualization of gesture patterns.

## Load the Dataset

First, we need to load the dataset. If you're using TensorFlow Datasets (`tfds`), you can load the dataset as follows:

```Python
import tensorflow_datasets as tfds

# Load the SmartWatch Gestures Dataset
ds, ds_info = tfds.load('smartwatch_gestures', with_info=True, as_supervised=False)

# Get the training dataset
train_ds = ds['train']
```

## Explore the Dataset

We'll explore the dataset to understand the structure of the data, including the features available (such as accelerometer and gyroscope sensor readings) and the target variable (gesture labels).

### Data Exploration

Let's plot acceleration patterns for a specific gesture performed by different participants. This will help us see the variance in how different people perform the same gesture.

```Python
import matplotlib.pyplot as plt

def plot_accelerations(features, gesture_id, participant_id):
    # Extract acceleration data for the specified gesture and participant
    accel_x = []
    accel_y = []
    accel_z = []

    for row in features:
        if row['gesture'].numpy() == gesture_id and row['participant'].numpy() == participant_id:
            for feature in row['features']:
                accel_x.append(feature['accel_x'])
                accel_y.append(feature['accel_y'])
                accel_z.append(feature['accel_z'])

    # Plot the acceleration data
    plt.figure(figsize=(15, 5))
    plt.plot(accel_x, label='Accel X')
    plt.plot(accel_y, label='Accel Y')
    plt.plot(accel_z, label='Accel Z')
    plt.title(f'Acceleration Patterns for Gesture {gesture_id}, Participant {participant_id}')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()

# Example: Plot accelerations for gesture 0 performed by participant 1
plot_accelerations(train_ds, gesture_id=0, participant_id=1)
```

### Convert to Pandas DataFrame for Easier Analysis

For easier analysis, we'll convert the TensorFlow dataset to a Pandas DataFrame. This step might involve batch processing and concatenation since TensorFlow datasets are typically in a format optimized for machine learning rather than data analysis.

```Python
import pandas as pd
import tensorflow as tf

# Load the dataset
train_ds = tfds.load('smartwatch_gestures', split='train')

# Initialize lists to hold data
attempt, accel_x, accel_y, accel_z, time_event, time_millis, time_nanos, gesture, participant = ([] for _ in range(9))

# Extract data from the dataset
for row in train_ds:
    attempt.extend(row['attempt'].numpy())
    gesture.extend(row['gesture'].numpy())
    participant.extend(row['participant'].numpy())
    
    # Extract features from each sequence
    for feature in row['features']:
        accel_x.extend(feature['accel_x'].numpy())
        accel_y.extend(feature['accel_y'].numpy())
        accel_z.extend(feature['accel_z'].numpy())
        time_event.extend(feature['time_event'].numpy())
        time_millis.extend(feature['time_millis'].numpy())
        time_nanos.extend(feature['time_nanos'].numpy())

# Create a DataFrame
df = pd.DataFrame({
    'Attempt': attempt,
    'Accel_X': accel_x,
    'Accel_Y': accel_y,
    'Accel_Z': accel_z,
    'Time_Event': time_event,
    'Time_Millis': time_millis,
    'Time_Nanos': time_nanos,
    'Gesture': gesture,
    'Participant': participant
})

# Display the DataFrame
print(df)
```

## Basic Statistics and Visualization

With the data in a Pandas DataFrame, we can perform basic statistics to understand the sensor data distributions and visualize some gestures.

## Next steps

1. **Preprocess:** Normalize or standardize the accelerometer data to ensure that each feature contributes equally to the model's learning
2. **Split:** Split the dataset into training and testing sets to evaluate the model's performance. You can use `**train_test_split**` from `**sklearn.model_selection**` for this purpose
3. **Define a model:**
    - **Machine Learning Models**: You can start with traditional machine learning models such as Random Forest, Support Vector Machines (SVM), or k-Nearest Neighbors (k-NN).
    - **Deep Learning Models**: If you have sufficient data, you can use deep learning models such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) for sequence data.
4. **Train:** Train the selected model on the training dataset. Use the preprocessed accelerometer data as input features and the encoded gesture labels as target variables.
5. **Evaluate:** After training, evaluate the model's performance on the testing dataset. Use appropriate evaluation metrics such as accuracy, precision, recall, or F1-score to assess the model's performance.

### Example

```Python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Preprocessing (if needed)
# ...

# Split the data into training and testing sets
X = df[['Accel_X', 'Accel_Y', 'Accel_Z']]
y = df['Gesture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
rf_accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", rf_accuracy)


# Define the XGBoost model
x_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(df['Gesture'].unique()), random_state=42)

# Train
x_model.fit(X_train, y_train)

# Make predictions
y_pred = x_model.predict(X_test)

# Evaluate the model
x_accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", x_accuracy)

```

# Practice: Atrial Fibrillation

## Download the data

```Python
# Download the patient files
!wget -r -N -c -np https://physionet.org/files/afpdb/1.0.0/
```

## List the data files

```Python
import os

# Base path where the dataset was downloaded
base_path = 'physionet.org/files/afpdb/1.0.0/'

# Initialize lists to hold file paths for the control and PAF groups
control_files = []
paf_files = []

# Walk through the directory structure
for root, dirs, files in os.walk(base_path):
    for file in files:
        # Check if the file is a .hea file (indicating an ECG record header)
        if file.endswith('.hea'):
            # Construct the full path to the .hea and corresponding .dat file
            header_path = os.path.join(root, file)
            data_path = os.path.join(root, file.replace('.hea', '.dat'))
            
            # Check if the file belongs to the control group (prefix 'n') or the PAF group (prefix 'p')
            if file.startswith('n'):
                control_files.append((header_path, data_path))
            elif file.startswith('p'):
                paf_files.append((header_path, data_path))
```

## Load the data

```Python
import numpy as np

def read_ecg_record(header_path, data_path):
    # Read the header file to get metadata
    with open(header_path, 'r') as f:
        header = f.readlines()
    
    # Parse the header for necessary information
    # This is a simplified example; you may need to adjust parsing based on the actual header content
    num_samples = int(header[0].split(' ')[3])
    sample_rate = int(header[0].split(' ')[2])
    
    # Read the binary ECG data from the .dat file
    with open(data_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.int16)
    
    # Convert raw data into a DataFrame
    # This assumes 2-channel ECG data; adjust as necessary
    ecg_data = pd.DataFrame(raw_data.reshape((num_samples, -1)), columns=['Channel_1', 'Channel_2'])
    
    return ecg_data

def aggregate_patient_data(patient_files, patient_class):
    all_records = []
    
    for header_path, data_path in patient_files:
        record_df = read_ecg_record(header_path, data_path)
        record_df['Patient_Class'] = patient_class
        all_records.append(record_df)
    
    # Combine all records for this patient into a single DataFrame
    patient_df = pd.concat(all_records).reset_index(drop=True)
    
    return patient_df

# Aggregate data for control and PAF groups
control_data = aggregate_patient_data(control_files, 'Control')
paf_data = aggregate_patient_data(paf_files, 'PAF')

# Combine data from both groups into a single DataFrame
all_patient_data = pd.concat([control_data, paf_data]).reset_index(drop=True)
```

## Basic Statistics

Calculating some basic statistics such as the mean and standard deviation for each ECG channel will give us an idea of the overall signal levels and variability.

```Python
# Calculate basic statistics for each channel and patient class
basic_stats = all_patient_data.groupby('Patient_Class').agg({
    'Channel_1': ['mean', 'std'],
    'Channel_2': ['mean', 'std']
})

print(basic_stats)
```

This code snippet calculates the mean and standard deviation for `Channel_1` and `Channel_2` of the ECG signals, grouped by the patient class (Control or PAF). The `groupby` method is used to separate the data by patient class, and the `agg` method calculates the specified statistics for each group.

## Visualization

Visualizing the ECG signals can provide a more intuitive understanding of the differences between control and PAF patients. We can plot a small segment of the ECG signals for a visual comparison.

```Python
import matplotlib.pyplot as plt

def plot_ecg_samples(df, title, num_samples=1000):
    plt.figure(figsize=(15, 5))
    plt.plot(df['Channel_1'].iloc[:num_samples], label='Channel 1')
    plt.plot(df['Channel_2'].iloc[:num_samples], label='Channel 2', alpha=0.7)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Select a small sample from each class for plotting
control_sample = control_data.iloc[:1000]
paf_sample = paf_data.iloc[:1000]

# Plot the samples
plot_ecg_samples(control_sample, 'Control Group ECG Sample')
plot_ecg_samples(paf_sample, 'PAF Group ECG Sample')
```

The `plot_ecg_samples` function is defined to plot a specified number of samples from both ECG channels. It then selects a small segment from the control and PAF data and uses this function to create plots for each. This visual comparison can help in identifying any noticeable differences in the ECG signal patterns between the two groups.