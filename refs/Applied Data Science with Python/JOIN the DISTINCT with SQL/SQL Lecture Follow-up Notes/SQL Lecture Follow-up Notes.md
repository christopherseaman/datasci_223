# Working from Colab

It’s possible to do this exercise on Colab, but you’ll have to upload the chinook.sqlite database (download from GitHub, upload to Colab). When you open a notebook from GitHub on Colab it only opens the notebook, no other files come over.

![[IMG_0087.jpeg]]

# GitHub sync issues

There are a couple ways to address conflicts when you are trying to update your fork.

## Best practice

Keep your `main` branch clean and create a branch for each exercise when you work on it

## Resolving conflicts

There are three main ways to resolve conflicts:

1. (easiest) Create a branch of `main` to save your work, then discard commits on `main` when syncing
2. (outside of git) Copy the files you want to save outside the repo, discard commits when syncing, add the files back
3. (manual resolution) Choose with changes to keep from each fork. I find this easier in VS Code than on the GitHub website, but neither is really _easy_

# Querying dataframes

You can query dataframes a few different ways:

## `df.query()`

Pandas dataframes have a built-in SQL query method, but it is limited to filtering results similar to a `WHERE` clause:

```Python
# Query using %sql magic
res = %sql SELECT * FROM results

# Filter for only points-winners using .query()
res.query('points >= 1')
```

## `%sql` and SQLite’s `-- persist`

When using SQLite, we have to tell it about the dataframe for `%sql` to be able to query it. This can be done by creating temporary tables using the `-- persist` statement:

```Python
# Definte a dataframe, it could be from anywhere
%sql df << SELECT * FROM db_table;

# Register the DataFrame as a temporary SQL table

# First, drop the table from the database in case it exists
%sql DROP TABLE IF EXISTS my_table;

# Then, register `df` as temp table `my_table` within SQLite
%sql --persist df my_table

# Now you can query the DataFrame using SQL
%sql SELECT * FROM my_table;
```

## `%sql` and DuckDB w/ jupysql

We can use more fully featured libraries to query dataframes directly:

- `duckdb` instead of `sqlite`
- `jupysql` instead of `python-sql` (note: jupysql prints an ad when loaded)

> [!important]  
> Warning! ipython-sql and jupysql cannot co-exist in the same environment  

```Python
# Use this with the F1 dataset
# Install DuckDB and jupysql
%pip install -U jupysql duckdb-engine

# Load packages
import pandas as pd
import zipfile
import os
import duckdb

# Configure the SQL magic
%load_ext sql
%config SqlMagic.autopandas = True
%config SqlMagic.feedback = False
%config SqlMagic.displaycon = False

# Use %sql magic to connect to the DuckDB database
%sql duckdb:///:memory:

# Unzip the data
with zipfile.ZipFile('F1.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# Load each CSV into a DataFrame
tables = []
for f in os.listdir('data'):
    if f.endswith('.csv'):
        table_name = f.replace('.csv', '')
        file_path = os.path.join('data', f)
        # tables[table_name] = pd.read_csv(file_path)
        tables.append(table_name)
        globals()[table_name] = pd.read_csv(file_path)

%sql SELECT * FROM results LIMIT 5
```