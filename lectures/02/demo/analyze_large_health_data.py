import argparse

INPUT_CSV = "patients_large.csv"
OUTPUT_CSV = "summary.csv"

def analyze_with_polars():
    import polars as pl
    print("Using polars with lazy streaming...")
    lazy_df = pl.scan_csv(INPUT_CSV)
    result = (
        lazy_df
        .filter(pl.col("Age") > 65)
        .groupby("diagnosis")
        .agg(pl.count())
        .collect(streaming=True)
    )
    print(result)
    result.write_csv(OUTPUT_CSV)
    print(f"Saved summary to {OUTPUT_CSV}")

def analyze_with_pandas():
    import pandas as pd
    print("Using pandas (may fail on large files)...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except MemoryError:
        print("MemoryError: file too large for pandas on this machine.")
        return
    filtered = df[df['Age'] > 65]
    summary = filtered.groupby('diagnosis').size().reset_index(name='count')
    print(summary)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved summary to {OUTPUT_CSV}")

def main():
    parser = argparse.ArgumentParser(description="Analyze large health dataset with polars or pandas")
    parser.add_argument('--backend', choices=['polars', 'pandas'], default='polars',
                        help="Which library to use (default: polars)")
    args = parser.parse_args()

    if args.backend == 'polars':
        analyze_with_polars()
    else:
        analyze_with_pandas()

if __name__ == "__main__":
    main()