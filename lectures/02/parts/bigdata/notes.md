# Working with Data Bigger Than Memory 📊🐘

---

## Why Does This Matter?

- Health data can be **huge**: millions of patients, years of records
- Your laptop's RAM is **limited**
- Need tools and techniques to **analyze big data efficiently**

<!--
Big data challenges are common in health research, requiring scalable solutions.
-->

---

## Tools for Big Data

| Tool     | Key Features                                   | When to Use                          |
|----------|-----------------------------------------------|-------------------------------------|
| **pandas**  | Easy, powerful, in-memory                   | Small to medium datasets            |
| **polars**  | Fast, Rust-based, lazy evaluation           | Larger datasets, faster processing  |
| **spark**   | Distributed, scales to clusters             | Very large, distributed data        |
| **duckdb**  | SQL database, fast analytics                | SQL queries, medium-large data      |
| **dask**    | Parallel, pandas-like API                   | Larger-than-memory, parallel tasks  |

<!--
Choose tools based on data size, complexity, and available resources.
-->

---

## pandas vs polars 🐼⚡

- **pandas 3.0** moving to **Arrow** backend for better performance
- **polars** built on Arrow from the start, supports **lazy** and **streaming** APIs
- polars often **faster** and uses **less memory**
- Both have **similar syntax**, so switching is easy

Example:

```python
import polars as pl

df = pl.read_csv("bigfile.csv")
result = df.filter(pl.col("age") > 65).groupby("diagnosis").count()
```

<!--
polars enables faster, scalable data processing with familiar syntax.
-->

---

## polars vs pandas Syntax Quick Review 📝

| **Task**                     | **pandas**                                         | **polars**                                         |
|------------------------------|----------------------------------------------------|----------------------------------------------------|
| Import                      | `import pandas as pd`                              | `import polars as pl`                              |
| Read CSV                    | `pd.read_csv("file.csv")`                          | `pl.read_csv("file.csv")`                          |
| Lazy read CSV               | *not supported*                                   | `pl.scan_csv("file.csv")`                          |
| Select columns              | `df[['age', 'sex']]`                              | `df.select(['age', 'sex'])`                        |
| Filter rows                 | `df[df['age'] > 65]`                              | `df.filter(pl.col('age') > 65)`                    |
| Group by and aggregate      | `df.groupby('diagnosis').size()`                  | `df.groupby('diagnosis').count()`                  |
| Chain operations            | `df[df['age'] > 65].groupby('diagnosis').size()`  | `df.filter(pl.col('age') > 65).groupby('diagnosis').count()` |
| Save CSV                    | `df.to_csv("out.csv")`                            | `df.write_csv("out.csv")`                          |

<!--
polars syntax is similar to pandas, but uses `pl.col()` for expressions and supports lazy queries.
-->

---

## When pandas Fails but polars Works 🚫🐼✅⚡

### Scenario

- You have a **20GB CSV** of patient records
- Your laptop has **8GB RAM**
- You want to filter and summarize the data

### pandas Attempt

```python
import pandas as pd

df = pd.read_csv("patients_20GB.csv")  # tries to load entire file
# Likely causes MemoryError or system freeze
```

- **Problem:** pandas loads the **whole file into memory**
- **Result:** Crashes or becomes unresponsive

### polars Solution

```python
import polars as pl

result = (
    pl.scan_csv("patients_20GB.csv")  # lazy, no full load
    .filter(pl.col("age") > 65)
    .groupby("diagnosis")
    .agg(pl.count())
    .collect(streaming=True)  # processes in chunks
)
```

- **Why it works:** polars **streams data in chunks**, never loads all at once
- **Result:** Finishes successfully, even on a laptop

<!--
polars' lazy and streaming execution enables analysis of datasets far larger than RAM, where pandas fails.
-->

---

## Using polars with Larger-than-Memory Data 🧠💾

### Lazy Evaluation

- polars can **build a query plan** without loading all data
- Only loads data **when needed** for the final result
- Use `.scan_csv()` instead of `.read_csv()` for big files

Example:

```python
import polars as pl

lazy_df = pl.scan_csv("bigfile.csv")
result = (
    lazy_df
    .filter(pl.col("age") > 65)
    .groupby("diagnosis")
    .agg(pl.count())
    .collect()  # triggers execution
)
```

<!--
Lazy queries let polars optimize and avoid loading unnecessary data.
-->

---

### Streaming Execution

- polars can **process data in chunks** (streaming)
- Avoids loading entire dataset into memory
- Enabled automatically with `.scan_csv()` and `.collect(streaming=True)`

Example:

```python
result = (
    pl.scan_csv("bigfile.csv")
    .filter(pl.col("age") > 65)
    .groupby("diagnosis")
    .agg(pl.count())
    .collect(streaming=True)
)
```

<!--
Streaming allows analysis of datasets much larger than RAM.
-->

---

### Chunked Reading

- For very large files, read in **chunks** and process incrementally
- Combine results after processing each chunk

Example:

```python
for batch in pl.read_csv("bigfile.csv", batch_size=100_000):
    # process each batch separately
    pass
```

<!--
Chunking is useful when lazy/streaming is not enough or for custom workflows.
-->

---

## Larger-than-Memory Operations

### What's Easy?

- **Parallelizable** tasks:
  - Calculating mean, sum, counts
  - Filtering rows independently
- **Scan-based** operations:
  - Reading data in chunks

### What's Hard?

- **Non-partitioned joins**
- **Complex groupbys**
- **Sorting entire datasets**

### Strategies

- **Chunking:** process data in pieces
- **Out-of-core algorithms:** work without loading all data at once
- **Distributed frameworks:** use multiple machines (e.g., Spark)

<!--
Understanding what scales well helps design efficient analyses.
-->

---

## Demo: Analyzing a Large Health Dataset with polars 🚀

### Goal

- Filter and summarize a large CSV of patient data **without running out of memory**

### Steps

1. Download or generate a large CSV (e.g., 1-10 million rows)
2. Use `pl.scan_csv()` to create a lazy dataframe
3. Filter patients over 65
4. Group by diagnosis, count patients
5. Collect results with `streaming=True`
6. Save summary to a new CSV

### Sample Code

```python
import polars as pl

result = (
    pl.scan_csv("patients_large.csv")
    .filter(pl.col("age") > 65)
    .groupby("diagnosis")
    .agg(pl.count())
    .collect(streaming=True)
)

result.write_csv("summary.csv")
```

### Expected Outcome

- A CSV file with diagnosis categories and patient counts
- Completed **without crashing your laptop**

<!--
Demo shows how polars handles big data efficiently with lazy and streaming.
-->

---

## Assignment: Big Data Health Analysis 🏥📈

### Task

- Use polars (or dask) to analyze a large health dataset (provided or generated)
- Perform:
  - Filtering (e.g., age, diagnosis)
  - Grouping and aggregation
  - Save results to a file

### Requirements

- Use **lazy evaluation** (`scan_csv`) and/or **streaming**
- Avoid loading entire dataset into memory
- Include code comments explaining steps
- Write a brief reflection:
  - Challenges faced
  - How polars helped
  - What you learned

### Bonus

- Try chunked reading and incremental processing
- Compare with pandas (if feasible)

<!--
Assignment builds practical skills for scalable health data analysis.
-->

---

## Resources

- [pandas documentation](https://pandas.pydata.org/)
- [polars documentation](https://pola.rs/)
- [dask documentation](https://dask.org/)
- [duckdb documentation](https://duckdb.org/)
- [Spark documentation](https://spark.apache.org/)