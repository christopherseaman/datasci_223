# Assignment 02 Hints

## Lazy pipeline tips

- Use `pl.scan_parquet` and chain `.select()` early to keep the plan lean.
- `LazyFrame.collect_schema()` is the fastest way to inspect dtypes.
- `LazyFrame.explain()` should show scans → filters → joins → aggregates.

## ICD-10 filtering

- `pl.col("code").str.slice(0, 3).is_in(prefixes)` is a safe way to use ICD prefixes.
- Keep the lookup table lean: select just `code` + `category` before joining.

## Prevalence math

- Build a `patients_with_dx` table with `.unique()` before joining back to patients.
- Use `fill_null(False)` after the join so every patient gets a boolean.

## Datetime helpers

- Use `pl.col("record_ts").str.strptime(pl.Datetime, strict=False)` before filtering.
- `datetime.fromisoformat(config["data"]["start_date"])` is a quick way to build the cutoff.

## Output checks

- Create output folders with `Path(...).parent.mkdir(parents=True, exist_ok=True)`.
- Read the Parquet output with `pl.read_parquet(...)` to sanity check row counts.

## Testing

- The autograder sets `POLARS_ASSIGNMENT_CONFIG` to point at a temp config file.
- If tests fail, run the notebook top-to-bottom and confirm the outputs exist.
