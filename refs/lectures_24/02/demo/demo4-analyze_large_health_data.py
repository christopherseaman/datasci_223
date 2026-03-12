import argparse
import time
import os
import sys

# Input files
INPUT_CSV = "patients_large.csv"
INPUT_PARQUET = "patients_large.parquet"
OUTPUT_CSV = "summary.csv"

def analyze_with_polars(format_type="parquet"):
    import polars as pl
    print(f"Using polars with lazy streaming on {format_type.upper()}...")
    
    # Choose input file based on format
    input_file = INPUT_PARQUET if format_type == "parquet" else INPUT_CSV
    
    # Time the operation
    start_time = time.time()
    print(f"Starting analysis at {time.strftime('%H:%M:%S')}")
    
    # Use appropriate scan function based on format
    if format_type == "parquet":
        lazy_df = pl.scan_parquet(input_file)
    else:
        lazy_df = pl.scan_csv(input_file)
    
    # Select only the columns we need to reduce memory usage
    lazy_df = lazy_df.select(["Age", "diagnosis"])
    
    result = (
        lazy_df
    #    .filter(pl.col("Age") > 65)
        .group_by("diagnosis")
        .agg(pl.len())
        .collect(engine='streaming')
    )
    
    elapsed_time = time.time() - start_time
    print(f"Analysis completed in {elapsed_time:.2f} seconds at {time.strftime('%H:%M:%S')}")
    print(result)
    
    result.write_csv(OUTPUT_CSV)
    print(f"Saved summary to {OUTPUT_CSV}")

def analyze_with_pandas(format_type="parquet"):
    import pandas as pd
    print(f"Using pandas with {format_type.upper()} (may fail on large files)...")
    
    # Choose input file based on format
    input_file = INPUT_PARQUET if format_type == "parquet" else INPUT_CSV
    
    # Time the operation
    start_time = time.time()
    print(f"Starting analysis at {time.strftime('%H:%M:%S')}")
    
    try:
        # Use appropriate read function based on format
        if format_type == "parquet":
            # With Parquet, we can select only the columns we need
            df = pd.read_parquet(input_file, columns=["Age", "diagnosis"])
        else:
            # With CSV, we need to read the whole file
            df = pd.read_csv(input_file)
            # Then select only the columns we need
            df = df[["Age", "diagnosis"]]
            
        filtered = df[df['Age'] > 65]
        summary = filtered.groupby('diagnosis').size().reset_index(name='count')
        
        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.2f} seconds at {time.strftime('%H:%M:%S')}")
        print(summary)
        
        summary.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved summary to {OUTPUT_CSV}")
    except MemoryError:
        print("MemoryError: file too large for pandas on this machine.")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze large health dataset with polars or pandas")
    parser.add_argument('--backend', choices=['polars', 'pandas'], default='polars',
                        help="Which library to use (default: polars)")
    parser.add_argument('--format', choices=['csv', 'parquet'], default='parquet',
                        help="Which file format to use (default: parquet)")
    args = parser.parse_args()

    if args.backend == 'polars':
        analyze_with_polars(args.format)
    else:
        analyze_with_pandas(args.format)

if __name__ == "__main__":
    main()