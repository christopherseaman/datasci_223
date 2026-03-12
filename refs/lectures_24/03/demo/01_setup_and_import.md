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

# Demo 1: Setup and Data Import

In this notebook, we'll set up our environment and import the NHANES data into DuckDB. This demo will show you how to:

1. Install and configure SQL tools
2. Set up DuckDB for data analysis
3. Import NHANES data from CSV files
4. Verify the data import


## Setup

First, let's install the required packages. We'll use:
- `jupysql` for SQL magic commands in Jupyter
- `duckdb-engine` for the DuckDB database engine
- `pandas` for data manipulation
- `polars` for efficient data processing

```python
%pip install jupysql duckdb-engine pandas polars --quiet
```

## Import Libraries and Configure SQL Magic

Now let's import the necessary libraries and configure SQL magic for our notebook:

```python
import pandas as pd
from sqlalchemy import create_engine
%load_ext sql

# Configure SQL magic for better output
%config SqlMagic.autocommit=True
%config SqlMagic.autopandas = True
%config SqlMagic.feedback = False
%config SqlMagic.displaycon = False
```

## Connect to DuckDB

DuckDB is an embedded database, which means it runs directly in your Python process without needing a separate server. This makes it perfect for data analysis:

```python
# Connect to DuckDB
# Execute SQL queries with pandas' `read_sql` function
# results_df = pd.read_sql("SELECT * FROM demographics LIMIT 5", engine)
engine = create_engine('duckdb:///nhanes.db')

# Use this engine for jupysql
%sql engine

# NOTE: memory db's are not shared between connections
# If using only jupysql, you can use the following:
# %sql duckdb:///:memory:
```

## Load Data

Let's load our NHANES data directly from CSV files. We'll use DuckDB's efficient CSV reader:

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

## Verify Data Import

Let's check that our data was imported correctly by looking at the first few rows of each table:

```python
%sql SELECT * FROM demographics LIMIT 5;
```

```python
%sql SELECT * FROM examination LIMIT 5;
```

```python
%sql SELECT * FROM laboratory LIMIT 5;
```

```python
%sql SELECT * FROM questionnaire LIMIT 5;
```

## Practice

Try these exercises:
1. Load another CSV file from the data directory
2. Use `DESCRIBE` to see the structure of each table
3. Count the number of rows in each table
4. Try loading a dataset of your own
