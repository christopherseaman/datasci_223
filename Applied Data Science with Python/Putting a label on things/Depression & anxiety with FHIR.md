# **Objective:**

Process raw FHIR data to extract patient demographic information, sleep patterns, activity levels, stress indicators, and the presence of depression or anxiety. Use this information to train a logistic regression model for predicting the risk of depression and anxiety.

# **FHIR Data Structure:**

Assume each patient's raw FHIR JSON object includes:

- A `Patient` resource with demographic information.
- `Observation` resources for sleep patterns (hours of sleep), activity levels (steps per day), and stress indicators (stress level on a scale of 1 to 10).
- A `Condition` resource indicating the presence or absence of depression and anxiety.

## **Tasks:**

1. Query examples walkthrough
    1. Query / load FHIR object
    2. Validate schema
    3. Load into `fhir.resources`
    4. Show dataset ready for ML training

  

---

1. Query relevant information from the raw FHIR JSON objects to create a pandas DataFrame.
2. Perform the necessary data preprocessing steps, including handling missing values and scaling.
3. Split the dataset into training and testing sets.
4. Train a logistic regression model on the training data.
5. Evaluate the model's performance on the test set.  
    Answer:  
    Below is the Python code to accomplish these tasks, adapted for the scenario of predicting depression and anxiety.