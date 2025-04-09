# Demo 0: Getting Started with Jupyter Notebooks 📓

---

## Goal

Learn to use Jupyter notebooks for interactive data exploration and analysis.

---

## Setup

- Ensure Jupyter is installed: `pip install jupyter`
- Launch Jupyter: `jupyter notebook`
- Create a new Python notebook

---

## Tasks

### 1. Basic Notebook Operations

1. **Create cells and run code**:
   - Create a new code cell
   - Enter a simple Python statement: `print("Hello, Health Data Science!")`
   - Run the cell with Shift+Enter or the Run button

2. **Use markdown for documentation**:
   - Create a markdown cell (change cell type to "Markdown")
   - Add a title, description, and bullet points:
   ```markdown
   # Patient Data Analysis
   
   This notebook explores patient vital signs data.
   
   Key metrics:
   - Heart rate
   - Blood pressure
   - Oxygen saturation
   ```

### 2. Using Magic Commands

1. **Install packages with `%pip`**:
   ```python
   %pip install pandas numpy matplotlib
   ```

2. **Measure execution time with `%timeit`**:
   ```python
   # Create a list of numbers
   numbers = list(range(1000000))
   
   # Measure time to calculate sum
   %timeit sum(numbers)
   ```

3. **Display plots inline with `%matplotlib`**:
   ```python
   %matplotlib inline
   import matplotlib.pyplot as plt
   
   # Create a simple plot
   plt.figure(figsize=(10, 6))
   plt.plot([0, 1, 2, 3, 4], [0, 3, 1, 5, 2])
   plt.title("Sample Patient Data")
   plt.xlabel("Time (hours)")
   plt.ylabel("Pain Level")
   plt.grid(True)
   plt.show()
   ```

### 3. Using Shell Commands

1. **List files in the current directory**:
   ```python
   !ls
   ```

2. **Check Python version**:
   ```python
   !python --version
   ```

3. **Create a directory and check it exists**:
   ```python
   !mkdir -p data
   !ls -la
   ```

### 4. Working with Data

1. **Create sample patient data**:
   ```python
   import pandas as pd
   import numpy as np
   
   # Create sample patient data
   np.random.seed(42)  # For reproducibility
   
   # Generate 100 patient records
   n_patients = 100
   
   data = {
       'patient_id': range(1, n_patients + 1),
       'age': np.random.randint(18, 90, size=n_patients),
       'heart_rate': np.random.normal(75, 15, size=n_patients).round().astype(int),
       'systolic_bp': np.random.normal(120, 20, size=n_patients).round().astype(int),
       'diastolic_bp': np.random.normal(80, 10, size=n_patients).round().astype(int),
       'temperature': np.random.normal(98.6, 1, size=n_patients).round(1),
       'o2_saturation': np.random.normal(97, 2, size=n_patients).round().astype(int)
   }
   
   # Create DataFrame
   patients_df = pd.DataFrame(data)
   
   # Display first few rows
   patients_df.head()
   ```

2. **Explore the data**:
   ```python
   # Basic statistics
   patients_df.describe()
   
   # Check for missing values
   patients_df.isna().sum()
   
   # Count patients by age group
   patients_df['age_group'] = pd.cut(patients_df['age'], 
                                    bins=[0, 30, 50, 70, 100], 
                                    labels=['<30', '30-50', '50-70', '>70'])
   patients_df['age_group'].value_counts()
   ```

3. **Visualize the data**:
   ```python
   # Create a histogram of heart rates
   plt.figure(figsize=(10, 6))
   plt.hist(patients_df['heart_rate'], bins=15, alpha=0.7)
   plt.title('Distribution of Patient Heart Rates')
   plt.xlabel('Heart Rate (bpm)')
   plt.ylabel('Number of Patients')
   plt.grid(True, alpha=0.3)
   plt.show()
   
   # Create a scatter plot
   plt.figure(figsize=(10, 6))
   plt.scatter(patients_df['age'], patients_df['systolic_bp'], alpha=0.7)
   plt.title('Age vs. Systolic Blood Pressure')
   plt.xlabel('Age (years)')
   plt.ylabel('Systolic BP (mmHg)')
   plt.grid(True, alpha=0.3)
   plt.show()
   ```

4. **Save the data**:
   ```python
   # Save to CSV
   patients_df.to_csv('data/patient_vitals.csv', index=False)
   
   # Verify the file was created
   !ls -la data/
   ```

---

## Expected Outcomes

- Students can create and run Jupyter notebooks
- Students can use markdown for documentation
- Students can use magic commands and shell commands
- Students can create, explore, and visualize simple datasets
- Students understand the interactive nature of notebooks

---

## Notes

<!--
Jupyter notebooks are ideal for health data science because they allow for:
- Interactive exploration of patient data
- Documentation of analysis steps
- Sharing of results with colleagues
- Reproducible research workflows
-->