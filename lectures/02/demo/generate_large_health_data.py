import pandas as pd
import numpy as np

# Source small dataset URL (synthetic diabetes data from UCI or similar)
SOURCE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Number of rows to generate
TARGET_ROWS = 5_000_000

# Output file
OUTPUT_CSV = "patients_large.csv"

def main():
    print("Loading source data...")
    df = pd.read_csv(SOURCE_URL)
    original_len = len(df)
    print(f"Original rows: {original_len}")

    reps = TARGET_ROWS // original_len + 1
    print(f"Replicating data {reps} times...")

    big_df = pd.concat([df] * reps, ignore_index=True)
    big_df = big_df.head(TARGET_ROWS)

    # Add some noise to age and glucose to avoid exact duplicates
    np.random.seed(42)
    big_df['Age'] = big_df['Age'] + np.random.randint(-5, 6, size=TARGET_ROWS)
    big_df['Glucose'] = big_df['Glucose'] + np.random.randint(-10, 11, size=TARGET_ROWS)

    # Clip to realistic ranges
    big_df['Age'] = big_df['Age'].clip(lower=0, upper=120)
    big_df['Glucose'] = big_df['Glucose'].clip(lower=0)

    # Add a fake diagnosis column
    diagnoses = ['Diabetes', 'Pre-diabetes', 'No Diabetes']
    big_df['diagnosis'] = np.random.choice(diagnoses, size=TARGET_ROWS, p=[0.3, 0.2, 0.5])

    print(f"Saving {TARGET_ROWS} rows to {OUTPUT_CSV}...")
    big_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()