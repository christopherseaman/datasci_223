---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# Demo 3: Complex SQL Queries

In this notebook, we'll explore more advanced SQL concepts using our NHANES data. We'll cover:

1. Common Table Expressions (CTEs)
2. Window Functions
3. Advanced JOIN Operations
4. Subqueries
5. Performance Optimization


## Setup

```python
%pip install jupysql duckdb-engine pandas polars --quiet
```

```python
from sqlalchemy import create_engine
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

## Common Table Expressions (CTEs)

CTEs help make complex queries more readable and maintainable:

```sql
-- Calculate BMI categories using CTE
WITH bmi_categories AS (
    SELECT 
        d.SEQN,
        d.RIDAGEYR,
        d.RIAGENDR,
        e.BMXBMI,
        CASE
            WHEN e.BMXBMI < 18.5 THEN 'Underweight'
            WHEN e.BMXBMI < 25 THEN 'Normal'
            WHEN e.BMXBMI < 30 THEN 'Overweight'
            ELSE 'Obese'
        END AS bmi_category
    FROM demographics d
    JOIN examination e ON d.SEQN = e.SEQN
)
SELECT 
    RIAGENDR,
    bmi_category,
    COUNT(*) AS count,
    AVG(RIDAGEYR) AS avg_age
FROM bmi_categories
GROUP BY RIAGENDR, bmi_category
ORDER BY RIAGENDR, bmi_category;
```

## Window Functions

Window functions allow us to perform calculations across rows related to the current row:

```sql
-- Calculate running average of BMI by age
SELECT 
    SEQN,
    RIDAGEYR,
    RIAGENDR,
    BMXBMI,
    AVG(BMXBMI) OVER (PARTITION BY RIAGENDR ORDER BY RIDAGEYR) AS running_avg_bmi
FROM (
    SELECT d.SEQN, d.RIDAGEYR, d.RIAGENDR, e.BMXBMI
    FROM demographics d
    JOIN examination e ON d.SEQN = e.SEQN
) AS combined
ORDER BY RIAGENDR, RIDAGEYR;
```

## Advanced JOIN Operations

Let's explore different types of JOINs and their use cases:

```sql
-- INNER JOIN: Get participants with complete data
SELECT d.SEQN, d.RIDAGEYR, d.RIAGENDR, e.BMXBMI, l.LBXTC
FROM demographics d
INNER JOIN examination e ON d.SEQN = e.SEQN
INNER JOIN laboratory l ON d.SEQN = l.SEQN
WHERE e.BMXBMI IS NOT NULL AND l.LBXTC IS NOT NULL
LIMIT 10;
```

```sql
-- LEFT JOIN: Get all participants with available data
SELECT d.SEQN, d.RIDAGEYR, d.RIAGENDR, e.BMXBMI, l.LBXTC
FROM demographics d
LEFT JOIN examination e ON d.SEQN = e.SEQN
LEFT JOIN laboratory l ON d.SEQN = l.SEQN
LIMIT 10;
```

## Subqueries

Subqueries help break down complex problems:

```sql
-- Find participants with above average BMI
SELECT SEQN, RIDAGEYR, RIAGENDR, BMXBMI
FROM (
    SELECT d.SEQN, d.RIDAGEYR, d.RIAGENDR, e.BMXBMI
    FROM demographics d
    JOIN examination e ON d.SEQN = e.SEQN
) AS combined
WHERE BMXBMI > (SELECT AVG(BMXBMI) FROM examination WHERE BMXBMI IS NOT NULL)
ORDER BY BMXBMI DESC
LIMIT 10;
```

## Performance Optimization

Let's look at some performance optimization techniques:

```sql
-- Use appropriate indexes
CREATE INDEX idx_demographics_seqn ON demographics(SEQN);
CREATE INDEX idx_examination_seqn ON examination(SEQN);
CREATE INDEX idx_laboratory_seqn ON laboratory(SEQN);
```

```sql
-- Create a view for health summary
-- NOTE: DuckDB does not support materialized views
CREATE VIEW health_summary AS
SELECT 
    d.SEQN,
    d.RIDAGEYR,
    d.RIAGENDR,
    e.BMXBMI,
    l.LBXTC,
    q.DIQ010
FROM demographics d
LEFT JOIN examination e ON d.SEQN = e.SEQN
LEFT JOIN laboratory l ON d.SEQN = l.SEQN
LEFT JOIN questionnaire q ON d.SEQN = q.SEQN;

-- Query the view
-- NOTE: This is done on the fly by querying the underlying tables
SELECT * FROM health_summary LIMIT 5;
```

## Practice

Try these exercises:
1. Create a CTE to analyze blood pressure trends by age group
2. Use window functions to find the top 10% of participants by BMI in each age group
3. Write a query to find participants with multiple health risk factors
4. Optimize a complex query using materialized views
