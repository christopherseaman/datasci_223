# Assignment 02 Tips

## Lazy scans

- Use `pl.scan_parquet(config["data"]["events_path"])` for events.
- Call `.collect_schema()` to verify columns without loading data.

## Polars patterns

- Convert `event_ts` to `Datetime` once in a `.with_columns(...)` call.
- For distinct patients per site, use `.select([...]).unique().group_by(...).agg(pl.len())`.
- Use `.str.starts_with("E11")` or the config prefix for ICD-10 filters.

Example pattern:

```python
# Verify parsing + filter logic before any joins/aggregations
events_filtered = events.with_columns(
    pl.col("event_ts").str.strptime(pl.Datetime, strict=False)
).filter(pl.col("event_ts") >= start_date)

dx_events = events_filtered.filter(pl.col("record_type") == "ICD-10-CM")
print(events_filtered.collect_schema())
print(dx_events.collect_schema())
```

Another example:

```python
# Count distinct patients by record type (diagnosis vs procedure)
patients_by_type = (
    events_filtered
    .select(["record_type", "patient_id"])
    .unique()
    .group_by("record_type")
    .agg(pl.len().alias("patients_seen"))
)
```

## Diabetes filter

```python
prefix = config["data"]["diabetes_prefix"]
dx_diabetes = dx_events.filter(pl.col("code").str.starts_with(prefix))
```

## Join flow

- `events` already includes `site_id`; join `sites` only for names/types.
- Join `hcpcs_codes` to procedures on `code` to get the `group` column.

```python
proc_with_group = proc_events.join(
    hcpcs.select(["code", "group"]), on="code", how="left"
)
```

## Streaming tips

- `.collect(engine="streaming")` works well after filtering and grouping.
- Avoid `.sort()` until after aggregation to keep streaming efficient.

## Output writing

- Create `outputs/` before writing.
- Use config paths so tests can find files.

## Debugging

- Print `LazyFrame.explain()` if filters seem ignored.
- If outputs are empty, check your date filter and prefix filter first.
