# Assignment 02 Hints

## Lazy pipeline tips

- Use `pl.scan_csv` and chain `.select()` early to keep the plan lean.
- `LazyFrame.collect_schema()` is the fastest way to inspect dtypes.
- `LazyFrame.explain()` should show scans → filters → joins → aggregates.

## Joining safely

- Encounters can contain multiple rows per patient. Use `select(["patient_id", "facility"]).unique()` to avoid duplicate joins.
- If you see too many rows in the output, check that mapping first.

## Datetime helpers

- Use `pl.col("timestamp").str.strptime(pl.Datetime, strict=False)` before calling `.dt.year()` / `.dt.month()`.
- `datetime.fromisoformat(config["data"]["start_date"])` is a quick way to build the filter.

## Output checks

- Create output folders with `Path(...).parent.mkdir(parents=True, exist_ok=True)`.
- Read the Parquet output with `pl.read_parquet(...)` to sanity check row counts.

## Testing

- The autograder sets `POLARS_ASSIGNMENT_CONFIG` to point at a temp config file.
- If tests fail, run the notebook top-to-bottom and confirm the outputs exist.
