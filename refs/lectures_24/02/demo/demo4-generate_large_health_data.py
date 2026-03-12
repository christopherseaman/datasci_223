import polars as pl
import numpy as np
import time
import os
import argparse
from tqdm import tqdm

# Source small dataset (local file)
SOURCE_FILE = "demo4-diabetes.csv"

# Default number of rows to generate
DEFAULT_ROWS = 5_000_000

# Output files
OUTPUT_CSV = "patients_large.csv"
OUTPUT_PARQUET = "patients_large.parquet"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a large health dataset for analysis.')
    parser.add_argument('--size', type=int, default=DEFAULT_ROWS,
                        help=f'Number of rows to generate (default: {DEFAULT_ROWS})')
    parser.add_argument('--no-hash', action='store_true',
                        help='Omit the memory-intensive hash column (default: include hash)')
    args = parser.parse_args()
    
    target_rows = args.size
    include_hash = not args.no_hash
    print(f"Generating dataset with {target_rows:,} rows...")
    print(f"Hash column: {'Included' if include_hash else 'Omitted'}")
    
    # Check if the source file exists
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Source file '{SOURCE_FILE}' not found.")
        print("Please download the diabetes dataset and save it as 'demo4-diabetes.csv'.")
        return
    
    # Load the source data
    print("Loading source data...")
    source_df = pl.read_csv(SOURCE_FILE)
    original_len = len(source_df)
    print(f"Original rows: {original_len}")
    
    # Calculate repetitions needed
    reps = target_rows // original_len + 1
    print(f"Replicating data {reps} times...")
    
    # Generate data efficiently with polars
    start_time = time.time()
    
    # Create the large dataset using polars
    print("Creating large dataset...")
    with tqdm(total=reps, desc="Replicating data") as pbar:
        big_df = pl.concat([source_df] * reps)[:target_rows]
        pbar.update(reps)
    
    # Add noise and new columns
    np.random.seed(42)  # For reproducibility
    print("Adding noise and new columns...")
    
    # Determine number of steps for progress bar
    total_steps = 4 if include_hash else 3
    with tqdm(total=total_steps, desc="Processing columns") as pbar:
        # Add noise to Age and Glucose
        big_df = big_df.with_columns([
            (pl.col("Age") + pl.Series(np.random.randint(-5, 6, size=target_rows))).clip(0, 120).alias("Age")
        ])
        pbar.update(1)
        
        big_df = big_df.with_columns([
            (pl.col("Glucose") + pl.Series(np.random.randint(-10, 11, size=target_rows))).clip(0, None).alias("Glucose")
        ])
        pbar.update(1)
        
        # Add diagnosis column
        big_df = big_df.with_columns([
            pl.Series(np.random.choice(['Diabetes', 'Pre-diabetes', 'No Diabetes'], 
                                     size=target_rows, p=[0.3, 0.2, 0.5])).alias("diagnosis")
        ])
        pbar.update(1)
        
        # Add memory-intensive hash column if enabled
        if include_hash:
            big_df = big_df.with_columns([
                pl.Series([f"hash_{i}_{np.random.randint(1000000)}" for i in range(target_rows)]).alias("hash_value")
            ])
            pbar.update(1)
    
    # Save as Parquet
    print(f"Saving {target_rows} rows to {OUTPUT_PARQUET}...")
    big_df.write_parquet(OUTPUT_PARQUET)
    parquet_time = time.time() - start_time
    parquet_size = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)  # MB
    
    # Save as CSV for comparison
    print(f"Saving CSV for comparison...")
    start_time = time.time()
    big_df.write_csv(OUTPUT_CSV)
    csv_time = time.time() - start_time
    csv_size = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)  # MB
    
    # Compare formats
    print("\nFormat Comparison:")
    print(f"CSV size: {csv_size:.2f} MB, Parquet size: {parquet_size:.2f} MB")
    
if __name__ == "__main__":
    main() 