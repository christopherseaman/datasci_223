# 02a_lazy_join: Lazy execution, query plans, and streaming patterns

Goal: demonstrate Polars lazy execution mechanics—`query.explain()`, streaming vs in-memory collection, `.sink_parquet()`, and when streaming hits limits.

We'll use the wearable tables (`user_profile`, `sleep_diary`, `sensor_hrv`) to show:

1. **Query plan inspection** with `.explain()` (projection/predicate pushdown)
2. **Streaming vs eager collection** (memory differences)
3. **Sink patterns** (write without materializing)
4. **Streaming-hostile operations** (what forces in-memory fallback)

Data relationships:

- `user_profile.user_id` joins to `sleep_diary.user_id`
- `sensor_hrv.device_id` encodes the user id suffix (e.g., `DEV-00012` → `USER-00012`)

Strategy: aggregate the large sensor table first (reduce cardinality), *then* join to dimensions. This keeps the join grain manageable.

## Data setup

```python
from pathlib import Path
import polars as pl
import altair as alt
import time
from generate_demo_data import generate_dataset

Path("data").mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(parents=True, exist_ok=True)

sensor_dir = Path("data/sensor_hrv")
if not sensor_dir.exists() or not list(sensor_dir.glob("*.parquet")):
    # Use 1M rows for faster demo execution
    generate_dataset(rows=1_000_000, output_dir="data")
```

## Steps

```python
sensor_dir = Path("data/sensor_hrv")
required_files = [
    Path("data/sleep_diary.parquet"),
    Path("data/user_profile.parquet"),
]

missing_files = [p for p in required_files if not p.exists()]
sensor_parts = list(sensor_dir.glob("*.parquet"))

if missing_files or not sensor_parts:
    raise FileNotFoundError(
        "Missing demo data.\n"
        f"- Missing files: {', '.join(str(p) for p in missing_files) if missing_files else '(none)'}\n"
        f"- Sensor parts found: {len(sensor_parts)}\n"
        "Run: uv run python generate_demo_data.py --size small --output-dir data"
    )
```

### 1) Scan sources lazily (nothing executes yet)

```python
users = pl.scan_parquet("data/user_profile.parquet")
sleep = pl.scan_parquet("data/sleep_diary.parquet")
sensor_raw = pl.scan_parquet("data/sensor_hrv/*.parquet")

# These are LazyFrame objects—no data loaded yet
print(f"users type: {type(users)}")
print(f"sensor type: {type(sensor_raw)}")

# Schema preview without triggering expensive resolution
print("\nSensor schema:")
print(sensor_raw.collect_schema())
```

**Optional performance tip: providing schema for CSV scans**

For Parquet files, schema is embedded (no need to specify). For CSV files, providing dtypes avoids inference and speeds up scans:

```python
# Example: if scanning CSV instead of Parquet
# schema = {
#     "device_id": pl.Utf8,
#     "ts_start": pl.Datetime,
#     "heart_rate": pl.Int64,
#     "hrv_sdnn": pl.Float64,
#     # ... etc
# }
# sensor_csv = pl.scan_csv("data/sensor_hrv/*.csv", schema=schema)
```

### 1b) SQL vs Polars expressions (equivalent prefilter)

Same idea, two syntaxes. Both produce the same lazy result.

```python
sensor_expr = (
    sensor_raw
    .filter(pl.col("missingness_score") <= 0.35)
    .select(["device_id", "ts_start", "hrv_sdnn", "hrv_rmssd", "heart_rate", "steps"])
)

ctx = pl.SQLContext()
ctx.register("sensor_raw", sensor_raw)

sensor_sql = ctx.execute(
    """
    SELECT device_id, ts_start, hrv_sdnn, hrv_rmssd, heart_rate, steps
    FROM sensor_raw
    WHERE missingness_score <= 0.35
    """
)

print(sensor_expr.explain())
print(sensor_sql.explain())

# Use either version; both are lazy and equivalent.
sensor = sensor_sql
```

### 2) Build a lazy query (no execution until `.collect()`)

Derive keys, filter nighttime segments, aggregate to nightly summaries, then join to dimensions.

```python
# Derive user_id from device_id, extract date
sensor_keyed = sensor.with_columns(
    pl.concat_str([
        pl.lit("USER-"),
        pl.col("device_id").str.split("-").list.get(1)
    ]).alias("user_id"),
    pl.col("ts_start").dt.date().alias("date"),
)

# Filter nighttime windows (10pm - 6am) and good-quality data
night_segments = (
    sensor_keyed
    .filter(
        pl.col("ts_start").dt.hour().is_between(22, 23) |
        pl.col("ts_start").dt.hour().is_between(0, 6)
    )
    .select(["user_id", "date", "hrv_sdnn", "hrv_rmssd", "heart_rate", "steps"])
)

# Aggregate sensor data to one row per user per night
nightly = (
    night_segments
    .group_by(["user_id", "date"])
    .agg([
        pl.len().alias("num_segments"),
        pl.mean("heart_rate").alias("night_mean_hr"),
        pl.mean("hrv_sdnn").alias("night_mean_sdnn"),
        pl.mean("hrv_rmssd").alias("night_mean_rmssd"),
        pl.sum("steps").alias("night_steps"),
    ])
)

# Join sleep diary + nightly physiology + user demographics
joined = (
    sleep
    .join(nightly, on=["user_id", "date"], how="inner")
    .join(
        users.select(["user_id", "age", "gender", "occupation"]),
        on="user_id",
        how="left"
    )
)

# Final summary: group by occupation + gender
summary = (
    joined
    .group_by(["occupation", "gender"])
    .agg([
        pl.len().alias("n_nights"),
        pl.mean("sleep_efficiency").alias("avg_sleep_efficiency"),
        pl.mean("night_mean_sdnn").alias("avg_night_sdnn"),
        pl.corr("sleep_efficiency", "night_mean_sdnn").alias("corr_sleep_sdnn"),
    ])
    .sort(["occupation", "gender"])
)

print("✅ Query built (still lazy, nothing executed)")
```

### 3) Inspect the query plan with `.explain()`

The query optimizer shows you what will actually run: filter pushdown, column pruning, join strategies.

```python
print(summary.explain())
```

**What to look for:**

- `FILTER` operations pushed down early (before joins)
- `SELECT` showing only needed columns
- Join type (`INNER JOIN`, `LEFT JOIN`)
- Aggregate operations batched

**Key insight:** Polars reorders your operations for efficiency. Filters get pushed to the scan, unused columns get dropped.

More on this next week with SQL!

### 4a) Collect with streaming engine (memory-bounded)

```python
t0 = time.perf_counter()
out_streaming = summary.collect(engine="streaming")
t_streaming = time.perf_counter() - t0

print(f"Streaming collect: {out_streaming.height:,} rows in {t_streaming:.2f}s")
print(f"Result memory: {out_streaming.estimated_size('mb'):.2f} MB")
out_streaming.head(10)
```

### 4b) Collect with default engine (for comparison)

```python
t0 = time.perf_counter()
out_default = summary.collect()
t_default = time.perf_counter() - t0

print(f"Default collect: {out_default.height:,} rows in {t_default:.2f}s")
print(f"Result memory: {out_default.estimated_size('mb'):.2f} MB")
```

**Comparison:**

```python
bench = pl.DataFrame({
    "engine": ["streaming", "default"],
    "seconds": [t_streaming, t_default],
})

alt.Chart(bench).mark_bar().encode(
    x=alt.X("engine:N", title=None),
    y=alt.Y("seconds:Q", title="Execution time (s)"),
).properties(width=300, height=200, title="Streaming vs Default")
```

**Key insight:** For this query, streaming may be *slower* because the result is small and fits in memory. Streaming shines when intermediate tables are huge.

### 5) Write results with `.sink_parquet()` (no materialization)

Instead of `.collect()` then `.write_parquet()`, we can write directly to disk without bringing the DataFrame into Python.

```python
summary.sink_parquet("outputs/sleep_hrv_by_group_sink.parquet")
print("✅ Wrote via .sink_parquet() (never materialized in Python)")

# Verify the output
result = pl.read_parquet("outputs/sleep_hrv_by_group_sink.parquet")
print(f"Rows written: {result.height:,}")
result.head()
```

**When to use `.sink_parquet()`:**

- Final output is large and you don't need it in memory
- Running batch jobs where Python memory is limited
- Building ETL pipelines that just transform and store

### 6) Streaming-hostile operation: global sort

Some operations can't stream—they need to see all data at once. Example: sorting the *entire* joined table.

```python
# This forces a full materialize because global sort needs all rows
global_sorted = (
    joined
    .sort(["user_id", "date"])  # Global sort across all partitions
)

print("\n--- Global sort query plan ---")
print(global_sorted.explain())

# Try streaming—Polars will fall back to in-memory for the sort
t0 = time.perf_counter()
sorted_out = global_sorted.collect(engine="streaming")
t_sort = time.perf_counter() - t0

print(f"\nGlobal sort (streaming engine): {sorted_out.height:,} rows in {t_sort:.2f}s")
print(f"Memory used: {sorted_out.estimated_size('mb'):.2f} MB")
print("⚠️  Note: streaming engine falls back to in-memory for global operations")
```

**Alternative: partition sort (sort within groups, then optionally merge)**

Polars can sort within partitions efficiently. For pre-grouped data, you can sort each group independently (O(n log k) where k = group size) rather than globally (O(n log n)).

```python
# Sort within user_id groups (each user's nights chronologically)
# This is faster because each subgroup is smaller
partitioned_sort = (
    joined
    .sort(["user_id", "date"], maintain_order=True)
)

t0 = time.perf_counter()
part_sorted = partitioned_sort.collect(engine="streaming")
t_part = time.perf_counter() - t0

print(f"\nPartitioned sort (by user_id, date): {part_sorted.height:,} rows in {t_part:.2f}s")
print(f"vs global sort: {t_sort:.2f}s")
print(f"Speedup: {t_sort / t_part:.1f}x")

# Polars streaming engine already uses partitioned sort strategies internally
# when the sort key aligns with existing partitions
print("\n✅ Polars automatically uses partition-aware sorting when beneficial")
```

### 7) Visual: sleep efficiency vs nightly HRV (sample of joined data)

```python
sample = (
    joined
    .select(["sleep_efficiency", "night_mean_sdnn", "occupation", "gender"])
    .collect(engine="streaming")
    .sample(n=2000, shuffle=True)
)

alt.Chart(sample).mark_circle(opacity=0.3, size=30).encode(
    x=alt.X("sleep_efficiency:Q", title="Sleep Efficiency (%)"),
    y=alt.Y("night_mean_sdnn:Q", title="Nighttime HRV SDNN (ms)"),
    color=alt.Color("gender:N", legend=alt.Legend(title="Gender")),
    tooltip=["occupation", "gender", "sleep_efficiency", "night_mean_sdnn"],
).properties(width=650, height=350, title="Sleep Quality vs HRV (sample of 2000 nights)")
```

### 8) Write final summary output (both formats)

```python
out_streaming.write_parquet("outputs/sleep_hrv_by_group.parquet", compression="snappy")
out_streaming.write_csv("outputs/sleep_hrv_by_group.csv")

print("✅ Outputs written:")
print("  - outputs/sleep_hrv_by_group.parquet")
print("  - outputs/sleep_hrv_by_group.csv")
print("  - outputs/sleep_hrv_by_group_sink.parquet")
```

## Checkpoints

- You understand the difference between `scan_*` (lazy) and `read_*` (eager)
- You can interpret `.explain()` output (filter pushdown, projection, join strategy)
- You know when to use `.collect(engine="streaming")` vs `.collect()`
- You can use `.sink_parquet()` to avoid materializing large results
- You recognize streaming-hostile patterns (global sorts, many-to-many joins)

## Key takeaways

| Pattern | Use case | Memory behavior |
|---------|----------|-----------------|
| `.scan_*().collect()` | Small result, fits in RAM | Materialize in Python |
| `.scan_*().collect(engine="streaming")` | Large intermediate tables | Bounded memory, may be slower for small results |
| `.scan_*().sink_parquet()` | Large final output, no need in Python | Never materializes in Python |
| Global sort/pivot in streaming | Unavoidable | Engine falls back to in-memory |
