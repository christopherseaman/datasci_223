#!/usr/bin/env python3
"""Demo data generator for wearable HRV + sleep diary.

Creates synthetic wearable sensor summaries aligned to Baigutanova et al.
(Scientific Data 2025; Samsung Galaxy Active 2).

Outputs (Parquet) into `--output-dir`:
- `sensor_hrv/part-*.parquet` (chunked for out-of-core generation)
- `sleep_diary.parquet`
- `user_profile.parquet`

Usage:
    uv run python generate_demo_data.py --size small --output-dir data
    uv run python generate_demo_data.py --size large --output-dir data
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from faker import Faker


DATA_DICT_PATH = Path(__file__).parent / "data_dictionary.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wearable HRV demo data")
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--size",
        choices=["small", "large"],
        help="Dataset size preset (small=1M rows, large=10M rows)",
    )
    size_group.add_argument(
        "--rows",
        type=int,
        help="Exact number of sensor rows to generate (overrides --size)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for generated Parquet files",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=200_000,
        help="Rows per chunk when writing sensor parquet",
    )
    args = parser.parse_args()
    
    # Set default if neither size nor rows is specified
    if args.size is None and args.rows is None:
        args.size = "small"
    
    return args


def load_data_dictionary() -> dict:
    with open(DATA_DICT_PATH) as f:
        return yaml.safe_load(f)


def build_user_tables(
    num_users: int,
    start_date: datetime,
    duration_days: int,
    faker: Faker,
    data_dict: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    columns = data_dict["tables"]["user_profile"]["columns"]

    profiles: list[dict] = []
    for i in range(num_users):
        profiles.append(
            {
                "user_id": f"USER-{i:05d}",
                "age": faker.random_int(min=18, max=65),
                "gender": faker.random_element(columns["gender"]["values"]),
                "height_cm": round(faker.random.uniform(150, 200), 1),
                "weight_kg": round(faker.random.uniform(45, 120), 1),
                "occupation": faker.random_element(columns["occupation"]["values"]),
                "lifestyle_regularity": faker.random_element(
                    columns["lifestyle_regularity"]["values"]
                ),
                "exercise_freq_weekly": faker.random_int(min=0, max=7),
                "smoking_freq": faker.random_element(columns["smoking_freq"]["values"]),
                "alcohol_freq": faker.random_element(columns["alcohol_freq"]["values"]),
                "coffee_freq": faker.random_element(columns["coffee_freq"]["values"]),
            }
        )

    users_df = pl.DataFrame(profiles)

    diaries: list[dict] = []
    for row in users_df.iter_rows(named=True):
        user_id = row["user_id"]
        age = row["age"]

        base_bedtime = 23 - (age - 18) * 0.02
        base_sleep_duration = 8 - (age - 18) * 0.03

        for day_offset in range(duration_days):
            current_date = start_date.date() + timedelta(days=day_offset)

            bedtime_hour = int(
                np.clip(base_bedtime + faker.random.uniform(-1.5, 1.5), 20, 23)
            )
            sleep_latency = faker.random_int(0, 30) if faker.random_int(0, 9) < 3 else 0
            sleep_duration = float(
                np.clip(base_sleep_duration + faker.random.uniform(-2, 2), 4, 12)
            )
            waso_minutes = faker.random_int(0, 30) if faker.random_int(0, 9) < 3 else 0

            bedtime = datetime.combine(current_date, datetime.min.time()).replace(
                hour=bedtime_hour
            )
            fall_asleep = bedtime + timedelta(minutes=sleep_latency)
            wake_time = fall_asleep + timedelta(hours=sleep_duration)

            total_sleep = sleep_duration - (waso_minutes / 60)
            time_in_bed = (wake_time - bedtime).total_seconds() / 3600
            sleep_efficiency = (
                (total_sleep / time_in_bed) * 100 if time_in_bed > 0 else 85
            )

            diaries.append(
                {
                    "user_id": user_id,
                    "date": current_date,
                    "bedtime": bedtime.time(),
                    "fall_asleep_time": fall_asleep.time(),
                    "wake_time": wake_time.time(),
                    "waso_minutes": waso_minutes,
                    "sleep_duration": round(total_sleep, 2),
                    "sleep_efficiency": round(sleep_efficiency, 1),
                    "sleep_latency": round(float(sleep_latency), 1),
                }
            )

    sleep_df = pl.DataFrame(diaries)
    return users_df, sleep_df


def write_sensor_hrv_parts(
    *,
    users_df: pl.DataFrame,
    start_date: datetime,
    duration_days: int,
    target_rows: int,
    out_dir: Path,
    seed: int,
    chunk_rows: int,
    logger: logging.Logger,
) -> int:
    """Generate `target_rows` sensor rows and write chunked Parquet files."""

    rng = np.random.default_rng(seed)

    num_users = users_df.height
    user_age_years = users_df.get_column("age").to_numpy()
    user_weight_kg = users_df.get_column("weight_kg").to_numpy()
    user_exercise_freq = users_df.get_column("exercise_freq_weekly").to_numpy()
    device_ids = np.array([f"DEV-{i:05d}" for i in range(num_users)], dtype=object)

    start = np.datetime64(start_date.replace(hour=0, minute=0, second=0, microsecond=0))
    total_segments = duration_days * 288

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-user parameters (adds realistic between-person variability) ---
    #
    # Demos compute aggregates like mean_hr/night_mean_hr. If every row is drawn
    # from the same distribution, device-level summaries look too similar.
    # Here we introduce:
    # - between-person baseline differences (resting HR)
    # - between-person variability differences (some "stable", some "noisy")
    # - within-person day-to-day drift (so daily means aren't flat lines)
    user_hr_rest = (
        62.0
        + 0.22 * (user_age_years.astype(float) - 30.0)
        + 0.08 * (user_weight_kg.astype(float) - 70.0)
        - 1.4 * user_exercise_freq.astype(float)
        + rng.normal(0.0, 5.5, size=num_users)
    )
    user_hr_rest = np.clip(user_hr_rest, 45.0, 105.0)

    hr_var_class = rng.choice(
        np.array(["low", "medium", "high"], dtype=object),
        size=num_users,
        p=[0.25, 0.55, 0.20],
    )
    user_hr_segment_sigma = np.empty(num_users, dtype=float)
    user_hr_daily_sigma = np.empty(num_users, dtype=float)

    low_mask = hr_var_class == "low"
    med_mask = hr_var_class == "medium"
    high_mask = hr_var_class == "high"

    user_hr_segment_sigma[low_mask] = rng.uniform(2.0, 5.0, size=int(low_mask.sum()))
    user_hr_segment_sigma[med_mask] = rng.uniform(5.0, 10.0, size=int(med_mask.sum()))
    user_hr_segment_sigma[high_mask] = rng.uniform(
        10.0, 18.0, size=int(high_mask.sum())
    )

    user_hr_daily_sigma[low_mask] = rng.uniform(0.5, 2.0, size=int(low_mask.sum()))
    user_hr_daily_sigma[med_mask] = rng.uniform(2.0, 5.0, size=int(med_mask.sum()))
    user_hr_daily_sigma[high_mask] = rng.uniform(5.0, 10.0, size=int(high_mask.sum()))

    # Activity propensity: affects steps distribution and therefore daily mean HR.
    user_activity_scale = rng.lognormal(mean=0.0, sigma=0.35, size=num_users)
    user_activity_scale = user_activity_scale * (0.9 + 0.04 * user_exercise_freq)
    user_activity_scale = np.clip(user_activity_scale, 0.6, 2.2)

    # Sensitivity of HR to step counts (some people spike more than others).
    user_hr_step_sensitivity = rng.uniform(0.06, 0.14, size=num_users)

    # Day-to-day drift around the resting baseline (drives per-device mean_hr spread).
    daily_hr_offset = rng.normal(
        loc=0.0,
        scale=user_hr_daily_sigma[:, None],
        size=(num_users, duration_days),
    )

    # Simple circadian effect (lower at night, slightly higher in evening).
    circadian_hr_by_hour = np.array(
        [
            -10.0,  # 00
            -10.0,  # 01
            -10.0,  # 02
            -10.0,  # 03
            -9.0,  # 04
            -8.0,  # 05
            -6.0,  # 06
            -4.0,  # 07
            -2.0,  # 08
            0.0,  # 09
            0.0,  # 10
            0.0,  # 11
            1.0,  # 12
            1.0,  # 13
            0.0,  # 14
            0.0,  # 15
            0.0,  # 16
            0.0,  # 17
            2.0,  # 18
            2.0,  # 19
            2.0,  # 20
            1.0,  # 21
            0.0,  # 22
            -2.0,  # 23
        ],
        dtype=float,
    )

    rows_written = 0
    part_idx = 0

    while rows_written < target_rows:
        n = min(chunk_rows, target_rows - rows_written)

        user_idx = rng.integers(0, num_users, size=n)
        segment_idx = rng.integers(0, total_segments, size=n)
        day_idx = (segment_idx // 288).astype(int)

        ts_start = start + (segment_idx.astype("timedelta64[m]") * 5)
        ts_end = ts_start + np.timedelta64(5, "m")

        hour = (
            ts_start.astype("datetime64[h]") - ts_start.astype("datetime64[D]")
        ).astype(int)

        age = user_age_years[user_idx].astype(float)

        activity_factor = np.ones(n, dtype=float)
        day_mask = (hour >= 9) & (hour <= 17)
        eve_mask = (hour >= 18) & (hour <= 22)
        activity_factor[day_mask] = rng.uniform(1.2, 2.0, size=int(day_mask.sum()))
        activity_factor[eve_mask] = rng.uniform(1.0, 1.5, size=int(eve_mask.sum()))

        activity_factor = activity_factor * user_activity_scale[user_idx]
        activity_factor = np.clip(activity_factor, 0.6, 3.0)

        steps = (rng.uniform(0, 50, size=n) * activity_factor).astype(int)
        acc_magnitude = rng.uniform(0.05, 0.5, size=n) * activity_factor

        charging_mask = hour < 6
        if charging_mask.any():
            steps[charging_mask] = (steps[charging_mask] * 0.2).astype(int)
            acc_magnitude[charging_mask] = acc_magnitude[charging_mask] * 0.5

        hr = (
            user_hr_rest[user_idx]
            + daily_hr_offset[user_idx, day_idx]
            + circadian_hr_by_hour[hour]
            + 4.0 * (activity_factor - 1.0)
            + steps * user_hr_step_sensitivity[user_idx]
            + rng.normal(0.0, user_hr_segment_sigma[user_idx], size=n)
        )
        hr = np.clip(hr, 40, 220).round().astype(int)

        base_sdnn = 100 - (age - 18) * 0.5
        sdnn = np.clip(
            base_sdnn
            + rng.uniform(-30, 30, size=n)
            - 0.10 * (hr.astype(float) - user_hr_rest[user_idx]),
            20,
            200,
        )
        rmssd = np.clip(sdnn * 0.7 + rng.uniform(-10, 10, size=n), 10, 150)
        lf_hf = np.clip(1.0 + rng.uniform(-0.7, 1.0, size=n), 0.3, 3.0)

        bvp_mean = rng.uniform(0.3, 1.5, size=n)
        spo2 = rng.uniform(95, 100, size=n)
        eda_mean = rng.uniform(0.2, 2.0, size=n) * activity_factor
        skin_temp = rng.uniform(32.0, 36.0, size=n)

        missingness = rng.uniform(0.0, 0.4, size=n)
        if charging_mask.any():
            missingness[charging_mask] = np.clip(
                missingness[charging_mask]
                + rng.uniform(0.3, 0.7, size=int(charging_mask.sum())),
                0.0,
                1.0,
            )

        chunk_df = pl.DataFrame(
            {
                "device_id": device_ids[user_idx],
                "ts_start": ts_start,
                "ts_end": ts_end,
                "heart_rate": hr,
                "hrv_sdnn": sdnn.round(2),
                "hrv_rmssd": rmssd.round(2),
                "hrv_lf_hf_ratio": lf_hf.round(3),
                "bvp_mean": bvp_mean.round(3),
                "spo2": spo2.round(1),
                "eda_mean": eda_mean.round(3),
                "skin_temp": skin_temp.round(1),
                "acc_magnitude": acc_magnitude.round(3),
                "steps": steps,
                "missingness_score": missingness.round(3),
            }
        )

        part_path = out_dir / f"part-{part_idx:05d}.parquet"
        chunk_df.write_parquet(part_path, compression="snappy", row_group_size=100_000)

        rows_written += n
        part_idx += 1

        if rows_written % 1_000_000 == 0 or rows_written == target_rows:
            logger.info(
                "  wrote %s / %s sensor rows", f"{rows_written:,}", f"{target_rows:,}"
            )

    return rows_written


def generate_dataset(
    size: str | None = None,
    rows: int | None = None,
    output_dir: Path | str = Path("data"),
    seed: int = 42,
    chunk_rows: int = 200_000,
) -> dict:
    """Generate wearable HRV demo dataset.
    
    Args:
        size: Size preset - 'small' (1M rows) or 'large' (10M rows). Ignored if rows is set.
        rows: Exact number of sensor rows to generate. Overrides size if provided.
        output_dir: Directory for output Parquet files
        seed: Random seed for reproducibility
        chunk_rows: Rows per chunk when writing sensor parquet
        
    Returns:
        dict: Generation metadata including file paths and sizes
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    faker = Faker()
    faker.seed_instance(seed)

    data_dict = load_data_dictionary()
    
    if rows is not None:
        target_sensor_rows = rows
        size_label = f"{rows:,} rows"
    else:
        if size is None:
            size = "small"
        size_config = data_dict["tables"]["sensor_hrv"]["row_count_target"]
        target_sensor_rows = int(size_config[size])
        size_label = size

    # Scale users based on target rows
    if target_sensor_rows >= 5_000_000:
        num_users = 1000
    elif target_sensor_rows >= 1_000_000:
        num_users = 150
    else:
        num_users = max(50, int(target_sensor_rows / 10_000))
    
    duration_days = 35

    logger.info("Generating %s dataset:", size_label)
    logger.info("  - Users: %d", num_users)
    logger.info("  - Duration: %d days", duration_days)
    logger.info("  - Target sensor rows: %s", f"{target_sensor_rows:,}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime(2024, 1, 1)

    logger.info("Generating user profiles + sleep diaries...")
    users_df, sleep_df = build_user_tables(
        num_users, start_date, duration_days, faker, data_dict
    )

    logger.info("Writing sensor/HRV parquet parts...")
    sensor_dir = output_dir / "sensor_hrv"
    sensor_rows = write_sensor_hrv_parts(
        users_df=users_df,
        start_date=start_date,
        duration_days=duration_days,
        target_rows=target_sensor_rows,
        out_dir=sensor_dir,
        seed=seed,
        chunk_rows=chunk_rows,
        logger=logger,
    )

    logger.info("Writing sleep diary + user profiles...")
    sleep_write_path = output_dir / "sleep_diary.parquet"
    users_write_path = output_dir / "user_profile.parquet"
    sleep_df.write_parquet(sleep_write_path, compression="snappy")
    users_df.write_parquet(users_write_path, compression="snappy")

    sensor_bytes = sum(p.stat().st_size for p in sensor_dir.glob("*.parquet"))
    sensor_size_mb = sensor_bytes / (1024 * 1024)
    sleep_size_mb = sleep_write_path.stat().st_size / (1024 * 1024)
    users_size_mb = users_write_path.stat().st_size / (1024 * 1024)

    logger.info("Generation complete!")
    logger.info("Files written to: %s", output_dir)
    logger.info(" - sensor_hrv/: %s rows, %.1f MB", f"{sensor_rows:,}", sensor_size_mb)
    logger.info(
        " - sleep_diary.parquet: %s rows, %.1f MB",
        f"{sleep_df.height:,}",
        sleep_size_mb,
    )
    logger.info(
        " - user_profile.parquet: %s rows, %.1f MB",
        f"{users_df.height:,}",
        users_size_mb,
    )

    metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "size": size_label,
        "seed": seed,
        "target_sensor_rows": target_sensor_rows,
        "actual_sensor_rows": sensor_rows,
        "num_users": num_users,
        "duration_days": duration_days,
        "outputs": {
            "sensor_hrv_glob": str((sensor_dir / "*.parquet").as_posix()),
            "sleep_diary": str(sleep_write_path.as_posix()),
            "user_profile": str(users_write_path.as_posix()),
        },
        "file_sizes_mb": {
            "sensor_hrv": round(sensor_size_mb, 2),
            "sleep_diary": round(sleep_size_mb, 2),
            "user_profile": round(users_size_mb, 2),
        },
    }

    metadata_path = output_dir / "generation_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info("Metadata written to: %s", metadata_path)
    
    return metadata


def main() -> None:
    args = parse_args()
    generate_dataset(
        size=args.size,
        rows=args.rows,
        output_dir=args.output_dir,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
    )


if __name__ == "__main__":
    main()
