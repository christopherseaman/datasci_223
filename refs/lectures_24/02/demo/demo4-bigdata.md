# Demo: Analyzing Large Health Data with polars and pandas 🏥📊

## Goal

Compare polars and pandas when analyzing a large health dataset, and demonstrate the benefits of using Parquet format for efficient data storage and processing.

## Setup

1. Big picture

   First, check total system memory:
   ```bash
   free -h
   ```

   Then in a separate terminal, start monitoring Python processes:
   ```bash
   watch -n 1 'ps -o pid,ppid,rss,vsize,pmem,pcpu,comm -C python'
   ```

   Now run the demo steps:
   ```bash
   # Change format between csv/parquet, backend between polars/pandas,
   time python3 demo4-analyze_large_health_data.py --backend polars --format parquet
   ```

   All together:
   ```bash
   python demo4-generate_large_health_data.py;\ # --size 100000000
   free -h; \
   watch -n 1 'ps -o pid,ppid,rss,vsize,pmem,pcpu,comm -C python'; \
   time python3 demo4-analyze_large_health_data.py --backend polars --format parquet
   ```

2. Download the source data:
```bash
python download_diabetes_data.py
```

3. Generate the large dataset (optional size parameter):
```bash
python demo4-generate_large_health_data.py --size 1000000
```

4. Analyze with different backends and formats:
```bash
python demo4-analyze_large_health_data.py --backend polars --format parquet
```

## Tasks

1. **Download the source data:**

```bash
python download_diabetes_data.py
```

This will download the diabetes dataset and save it as `demo4-diabetes.csv`.

2. **Generate data:**

```bash
python demo4-generate_large_health_data.py
```

This will create both CSV and Parquet files and show a comparison of their sizes and processing times. The script adds a memory-intensive hash column to ensure pandas will run out of memory when using CSV format.

3. **Analyze with polars using Parquet (recommended):**

```bash
# Use the memory monitor to track memory usage
python demo4-analyze_large_health_data.py --backend polars --format parquet
```

4. **Analyze with polars using CSV:**

```bash
python demo4-analyze_large_health_data.py --backend polars --format csv
```

5. **Analyze with pandas using Parquet:**

```bash
python demo4-analyze_large_health_data.py --backend pandas --format parquet
```

6. **Analyze with pandas using CSV (will likely fail):**

```bash
python demo4-analyze_large_health_data.py --backend pandas --format csv
```

7. **Observe:**
   - Parquet files are significantly smaller than CSV files
   - polars should succeed quickly with either format
   - pandas will likely crash with CSV due to the memory-intensive hash column
   - Processing times are faster with Parquet
   - Memory usage is tracked and displayed for each operation

8. **Inspect output `summary.csv`**

## Expected Outcomes

- Students see how polars handles big data efficiently
- Students understand pandas' memory limitations
- Students learn about the benefits of Parquet format for large datasets
- Students learn to choose the right tool and format for big data
- Students see how memory usage differs between libraries and formats

## Why Parquet?

Parquet is a columnar storage format that offers several advantages for health data:

- **Smaller file size**: 2-4x smaller than CSV
- **Faster queries**: Only reads needed columns
- **Schema enforcement**: Ensures data consistency
- **Predicate pushdown**: Filters data before loading
- **Better compression**: Efficient for healthcare data patterns
- **Column pruning**: Can read only needed columns, reducing memory usage

## Memory Usage Comparison

This demo includes a memory-intensive hash column to demonstrate the differences in memory usage:

| Library | Format | Memory Usage | Performance |
|---------|--------|--------------|-------------|
| polars  | Parquet| Low          | Very Fast   |
| polars  | CSV    | Medium       | Fast        |
| pandas  | Parquet| Medium       | Medium      |
| pandas  | CSV    | Very High    | Likely Fails|

## Memory Monitoring

### Using watch and free Commands

You can monitor memory usage in real-time using the `watch` and `free` commands:

```bash
# Show total memory usage on the system
free -h
```

This will show you the memory usage of your system and the Python process running the analysis script.

### Using System Monitoring Tools

#### On Linux:

```bash
# Open a terminal and run the analysis script
python demo4-analyze_large_health_data.py --backend pandas --format csv

# In a separate terminal, monitor the process
ps -eo pid,ppid,%cpu,%mem,rss,command | grep python
```

Or use `top` in a separate terminal:

```bash
# In a separate terminal
top
# Press 'p' and enter the PID of your Python process
```

#### On macOS:

```bash
# In a separate terminal
top -pid $(pgrep -f "demo4-analyze_large_health_data.py")
```

#### On Windows:

Open Task Manager (Ctrl+Shift+Esc) and go to the "Processes" tab to monitor the Python process.

These system tools provide direct monitoring without requiring any additional scripts or dependencies.

## Notes

This demo highlights the practical benefits of:
1. polars' lazy and streaming execution for large datasets
   - Uses the new streaming engine for efficient memory usage
   - Processes data in chunks without loading everything into memory
2. Parquet's efficient storage and processing capabilities
3. Choosing the right tool and format for healthcare data analysis
4. Memory usage tracking to understand resource requirements

<!--
This demo highlights the practical benefits of polars' lazy and streaming execution for large datasets.
-->