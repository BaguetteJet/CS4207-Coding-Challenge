import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "./data"
PROCESSED_DIR = "./data/processed"

TEST_START, TEST_END = "2026-01", "2026-09" # train on remaining months

FEATURES = [
    "PULocationID",
    "DOLocationID",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]

def extract_features(df):
    pickup = pd.to_datetime(df["tpep_pickup_datetime"])

    hour = pickup.dt.hour
    weekday = pickup.dt.day_of_week
    month = pickup.dt.month

    return pd.DataFrame({
        "PULocationID": df["PULocationID"],
        "DOLocationID": df["DOLocationID"],

        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),

        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),

        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),

        "is_weekend": (weekday >= 5).astype(int),
    })

def load(path):
    # Input: trip data
    df = pd.read_parquet(path, columns=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
    ])

    # Target: trip duration in minutes
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

    # Drop obvious bad records
    df = df.dropna()
    df = df[df["duration_min"].between(1, 120)]

    # Extract features
    features = extract_features(df)
    features["duration_min"] = df["duration_min"]
    return features

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "yellow_tripdata_*.parquet")))
    splits = {"train": [], "test": []}

    # Load and split the data into sets
    for file in files:
        month = os.path.basename(file).replace("yellow_tripdata_", "").replace(".parquet", "")
        split = "test" if TEST_START <= month <= TEST_END else "train"
        splits[split].append(load(file))

    # Save processed data to disk
    for split, dataframes in splits.items():
        df = pd.concat(dataframes, ignore_index=True)
        df.to_parquet(os.path.join(PROCESSED_DIR,f"{split}.parquet"), index=False)
        print(f"Saved {split}: {len(df):,} trips")
