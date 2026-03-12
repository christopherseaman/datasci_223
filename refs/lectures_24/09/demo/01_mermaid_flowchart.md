# Demo 1: Mermaid Flowchart - Clinical Data Pipeline

**Goal:** Create a comprehensive flowchart representing a real-world clinical data analysis pipeline using Mermaid syntax.

## Step 1: Set Up Your Environment

1. Open the [Mermaid Live Editor](https://mermaid.live) in your browser
2. Clear the default content
3. We'll build our diagram step by step

**Note on Jupyter Notebooks:** This demo uses Markdown ` ```mermaid ... ``` ` blocks. If you convert this file to a Jupyter Notebook (e.g., using Jupytext) and the diagrams don't render automatically:
    * Then, for each Mermaid diagram, change its cell type from Markdown to Code and add `%%mermaid` at the very beginning of the cell, before the `graph TD;` line.

## Step 2: Basic Structure

Start with this basic framework:

```mermaid
graph TD;
    A[Raw EHR Data] --> B[Data Validation];
```

**Test it:** Paste this into the Mermaid Live Editor. You should see two boxes connected by an arrow.

## Step 3: Add Decision Points

Now let's add some complexity with decision diamonds:

```mermaid
graph TD;
    A[Raw EHR Data] --> B[Data Validation];
    B --> C{Data Quality Check};
    C -->|Pass| D[Data Cleaning];
    C -->|Fail| E[Generate Error Report];
    E --> F[Manual Review Required];
```

**What's happening:** The diamond shape `{}` creates decision points, and we can label the arrows with conditions.

## Step 4: Complete Clinical Pipeline

Replace your diagram with this comprehensive version:

```mermaid
graph TD;
    A[Raw EHR Data] --> B[Data Validation];
    B --> C{Data Quality Check};
    C -->|Pass| D[Data Cleaning];
    C -->|Fail| E[Generate Error Report];
    E --> F[Manual Review Required];
    D --> G[Feature Engineering];
    G --> H{Analysis Type};
    H -->|Descriptive| I[Summary Statistics];
    H -->|Predictive| J[Model Training];
    H -->|Diagnostic| K[Hypothesis Testing];
    I --> L[Visualization];
    J --> M[Model Validation];
    K --> N[Statistical Tests];
    L --> O[Clinical Report];
    M --> P{Model Performance};
    P -->|Good| Q[Deploy Model];
    P -->|Poor| R[Retrain Model];
    N --> O;
    Q --> S[Monitor Performance];
    R --> J;
    O --> T[Stakeholder Review];
    T --> U[Implementation];
```

## Step 5: Style It Up

Add some visual flair with different node shapes:

```mermaid
graph TD;
    A[(EHR Database)] --> B[Data Validation];
    B --> C{Data Quality Check};
    C -->|Pass| D(Data Cleaning);
    C -->|Fail| E>Error Report];
    E --> F[[Manual Review]];
    D --> G(Feature Engineering);
    G --> H{Analysis Type};
    H -->|Descriptive| I[Summary Statistics];
    H -->|Predictive| J[Model Training];
    H -->|Diagnostic| K[Hypothesis Testing];
    I --> L[Visualization];
    J --> M[Model Validation];
    K --> N[Statistical Tests];
    L --> O((Clinical Report));
    M --> P{Model Performance};
    P -->|Good| Q[Deploy Model];
    P -->|Poor| R[Retrain Model];
    N --> O;
    Q --> S[Monitor Performance];
    R --> J;
    O --> T[Stakeholder Review];
    T --> U[Implementation];
```

**Node shapes used:**
- `[()]` - Database cylinder
- `()` - Rounded rectangle for processes
- `>]` - Asymmetric shape for outputs
- `[[]]` - Subroutine box
- `(())` - Circle for final outputs

## Step 6: Save Your Work

1. Click the "Actions" menu in Mermaid Live Editor
2. Choose "Download SVG" to save as a vector image
3. Copy the code to save in your notes

## Success Validation

Your final diagram should:
- ✅ Show the complete data flow from raw data to implementation
- ✅ Include decision points with labeled paths
- ✅ Use different shapes to indicate different types of operations
- ✅ Be readable and logically organized

## Bonus Challenge

Create a second diagram showing what happens when your model goes into production and starts making predictions that are... questionable:

```mermaid
graph TD;
    A[Model in Production] --> B[New Patient Data];
    B --> C[Model Prediction];
    C --> D{Prediction Reasonable?};
    D -->|Yes| E[Clinical Decision Support];
    D -->|"Patient age: -47 years"| F[Panic Mode];
    F --> G[Check Data Pipeline];
    G --> H[Find Bug in ETL];
    H --> I[Fix Bug];
    I --> J[Retrain Model];
    J --> K[Deploy Fix];
    K --> L[Update Resume];
```

This bonus diagram captures the reality of production ML systems in healthcare! 🎭