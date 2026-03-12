---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# SQL Basics for Health Data Science

<!--- 
This notebook introduces SQL basics using real health data from NHANES. We'll cover:
- Basic SELECT queries
- Filtering and sorting
- Working with multiple tables
- Common health data analysis patterns
--->


## Setup and Imports
%pip install jupysql duckdb-engine pandas polars --quiet

```python
from sqlalchemy import create_engine
import json
import pandas as pd

%load_ext sql

# Configure SQL magic for better output
%config SqlMagic.autopandas = True
%config SqlMagic.autocommit=True
%config SqlMagic.feedback = False
%config SqlMagic.displaycon = False

# Connect to DuckDB
# Execute SQL queries with pandas' `read_sql` function
# results_df = pd.read_sql("SELECT * FROM demographics LIMIT 5", engine)
engine = create_engine('duckdb:///:memory:')

# Use this engine for jupysql
%sql engine

# NOTE: memory db's are not shared between connections
# If using only jupysql, you can use the following:
# %sql duckdb:///:memory:
```

## Load Data

Let's load our NHANES data directly from CSV files:

```sql
-- Clean up any existing tables
DROP TABLE IF EXISTS questionnaire;
DROP TABLE IF EXISTS laboratory;
DROP TABLE IF EXISTS examination;
DROP TABLE IF EXISTS demographics;

-- Load demographics
CREATE TABLE demographics AS
SELECT * FROM read_csv_auto('data/demographics.csv');

-- Load examination
CREATE TABLE examination AS
SELECT * FROM read_csv_auto('data/examination.csv');

-- Load laboratory
CREATE TABLE laboratory AS
SELECT * FROM read_csv_auto('data/labs.csv');

-- Load questionnaire
CREATE TABLE questionnaire AS
SELECT * FROM read_csv_auto('data/questionnaire.csv');
```

# Show the Data Dictionary

```sql magic_args="data_dictionary << "
-- Show data dictionary for all tables
SELECT 
    'demographics' as table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'demographics'
UNION ALL
SELECT 
    'examination' as table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'examination'
UNION ALL
SELECT 
    'laboratory' as table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'laboratory'
UNION ALL
SELECT 
    'questionnaire' as table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'questionnaire'
ORDER BY table_name, column_name;
```

```python
display(data_dictionary)
```

```python
# Or use the one I made for you, 'nhanes_data_dictionary.json'

# Load the data dictionary
with open('nhanes_data_dictionary.json', 'r') as f:
    data_dict = json.load(f)

# Convert to DataFrame
df_dict = []
for table, columns in data_dict.items():
    for col, desc in columns.items():
        df_dict.append({
            'Table': table,
            'Column': col,
            'Description': desc
        })

# Display as DataFrame
pd.DataFrame(df_dict)
```

## Basic SELECT Queries

Let's start with some simple SELECT queries:

```sql
-- Get first 5 rows from demographics
SELECT * 
FROM demographics 
LIMIT 5;
```

## Filtering Data

```sql
-- Get participants over 60 years old
SELECT SEQN, RIDAGEYR, RIAGENDR
FROM demographics
WHERE RIDAGEYR > 60
ORDER BY RIDAGEYR DESC
LIMIT 10;
```

## Joining Tables

```sql
-- Combine demographics with blood pressure measurements
SELECT d.SEQN, d.RIDAGEYR, d.RIAGENDR,
       e.BPXSY1, e.BPXDI1
FROM demographics d
JOIN examination e ON d.SEQN = e.SEQN
WHERE d.RIDAGEYR > 60
LIMIT 10;
```

## Aggregation and Grouping

```sql
-- Average blood pressure by age group
SELECT 
    CASE 
        WHEN RIDAGEYR < 30 THEN '18-29'
        WHEN RIDAGEYR < 40 THEN '30-39'
        WHEN RIDAGEYR < 50 THEN '40-49'
        WHEN RIDAGEYR < 60 THEN '50-59'
        ELSE '60+'
    END as age_group,
    AVG(BPXSY1) as avg_systolic,
    AVG(BPXDI1) as avg_diastolic,
    COUNT(*) as n_participants
FROM demographics d
JOIN examination e ON d.SEQN = e.SEQN
GROUP BY age_group
ORDER BY age_group;
```

## Practice

Try these exercises:
1. Find the average BMI by gender
2. Count the number of participants in each age group
3. Calculate the percentage of participants with high blood pressure
4. Find the correlation between age and blood pressure
