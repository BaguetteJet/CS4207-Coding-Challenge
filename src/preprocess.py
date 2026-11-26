# preprocess.py - Igor Kochanski 23358459

# Files are read and sharded into equal smaller chunks
# Shards are multi-processed in parallel, then merged together (data parallelism)
# Preprocessing includes cleaning and feature extraction
# Feature extraction is multi-threaded (task parallelism)
# Worker 1 = sequential processing, Worker N = parallel processing
# Results are verified to be identical between sequential and parallel
# Work is done saved to a temporary directory and done in batches to reduce memory usage

import glob
import os
import csv
import argparse
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor # threads for feature extraction
from concurrent.futures import ProcessPoolExecutor # processes for sharding
from concurrent.futures import as_completed
from pathlib import Path
from tqdm import tqdm
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = "./data"
PROCESSED_DIR = "./data/processed"
RESULTS_DIR = "./results/preprocess"

SHARD_ROWS = 500_000 # rows per shard
WORKERS = 8 # number of processes for sharding
THREADS = 4 # number of threads for feature extraction

NON_DETERMINISTIC = False # if True, results may come back in a different order than the input files

# months before TEST_START are training data, months after TEST_END are ignored
TEST_START, TEST_END = "2026-01", "2026-09"

PICKUP, DROPOFF = "tpep_pickup_datetime", "tpep_dropoff_datetime"
COLUMNS = [
    PICKUP,
    DROPOFF,
    "trip_distance",
    "fare_amount",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
]

# Reasonable bounds for NYC taxi trips (same thresholds as example parallel_features.py).
MIN_DURATION, MAX_DURATION = 1, 180      # minutes
MIN_DISTANCE, MAX_DISTANCE = 0.1, 60     # miles
MIN_FARE, MAX_FARE = 3, 250              # dollars

# Fixed categories keep the train and test encodings identical.
ZONES = range(1, 266)                    # TLC taxi zone IDs
RATE_CODES = [0, 1, 2, 3, 4, 5, 6, 99]   # 0 = Flex Fare (null RatecodeID)

FEATURES = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


def month_of(path):
    # '.../yellow_tripdata_2026-03.parquet' -> '2026-03'
    return Path(path).stem.removeprefix("yellow_tripdata_")


def cyclic(values, period, name):
    # periodic feature encoding
    angle = 2 * np.pi * values / period
    return {f"{name}_sin": np.sin(angle), f"{name}_cos": np.cos(angle)}


def extract_features(df, threads=4):
    # task parallelism for feature extraction
    pickup = pd.to_datetime(df[PICKUP])

    hour = pickup.dt.hour
    weekday = pickup.dt.day_of_week
    month = pickup.dt.month
    year = pickup.dt.year

    tasks = [
        lambda: {
            "PULocationID": pd.Categorical(df["PULocationID"], categories=ZONES),
            "DOLocationID": pd.Categorical(df["DOLocationID"], categories=ZONES),
            "RatecodeID": pd.Categorical(df["RatecodeID"].fillna(0), categories=RATE_CODES),
        },
        lambda: {**cyclic(hour, 24, "hour")},
        lambda: {**cyclic(weekday, 7, "weekday"), "is_weekend": (weekday >= 5).astype(int)},
        lambda: {**cyclic(month - 1, 12, "month")},
    ]

    with ThreadPoolExecutor(threads) as executor:
        parts = executor.map(lambda task: task(), tasks)

    columns = {name: col for part in parts for name, col in part.items()}

    # year is kept for sampling, not a model feature
    return pd.DataFrame({**columns, "year": year}, index=df.index)[FEATURES + ["year"]]


def clean(df, path):
    # keep valid trips only, add duration_min column
    start = pd.Timestamp(f"{month_of(path)}-01")
    end = start + pd.DateOffset(months=1)

    pickup = pd.to_datetime(df[PICKUP])
    dropoff = pd.to_datetime(df[DROPOFF])
    duration = (dropoff - pickup).dt.total_seconds() / 60

    # no spill-over pickups, nulls, zero/negative durations, zero-distance trips, negative fares, or meter errors
    ok = (
        (pickup >= start)
        & (pickup < end)
        & duration.between(MIN_DURATION, MAX_DURATION)
        & df["trip_distance"].between(MIN_DISTANCE, MAX_DISTANCE)
        & df["fare_amount"].between(MIN_FARE, MAX_FARE)
        & df["RatecodeID"].fillna(0).isin(RATE_CODES)
        & df["PULocationID"].between(1, 265)
        & df["DOLocationID"].between(1, 265)
    )
    return df[ok].assign(duration_min=duration[ok])


def iter_shards(files):
    # yield fixed-size (shard_id, path, batch) shards, in file order
    shard_id = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=SHARD_ROWS, columns=COLUMNS, use_threads=False):
            yield shard_id, path, batch
            shard_id += 1

def process_shard(shard_id, path, batch, temp_dir):
    # clean one shard and write it to temp_dir, then return (file, rows_in, rows_out)
    df = clean(batch.to_pandas(), path)
    features = extract_features(df, threads=THREADS).assign(duration_min=df["duration_min"])

    out = os.path.join(temp_dir, f"shard_{shard_id:03d}.parquet")
    features.to_parquet(out, index=False)
    return out, batch.num_rows, len(df)


# DETERMINISTIC: results come back in shard order, not completion order
def process_parallel(shards, workers, temp_dir):

    if NON_DETERMINISTIC:
        return process_parallel_non_deterministic(shards, workers, temp_dir)

    # process shards in parallel, results come back in shard order
    results, pending = [], deque() # queue instead of as_completed() to preserve order of shards

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for shard in tqdm(shards, desc="Parallel", unit=" shard", leave=False):
            pending.append(executor.submit(process_shard, *shard, temp_dir))
            if len(pending) >= workers:
                results.append(pending.popleft().result())
        results.extend(future.result() for future in pending)

    return results


# NON-DETERMINISTIC: results come back in completion order, not shard order
def process_parallel_non_deterministic(shards, workers, temp_dir):
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_shard, *shard, temp_dir) for shard in shards]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel", unit=" shard", leave=False):
            results.append(future.result())

    return results


def merge_shards(shard_paths, output_path):
    # append the shards to one Parquet file, one shard in memory at a time
    schema = pq.ParquetFile(shard_paths[0]).schema_arrow

    with pq.ParquetWriter(output_path, schema) as writer:
        for path in tqdm(shard_paths, desc="Merging", unit=" shard", leave=False):
            writer.write_table(pq.ParquetFile(path).read())


def assert_parquet_equal(left_path, right_path):
    # compare in batches, so both full frames never coexist in memory
    left, right = pq.ParquetFile(left_path), pq.ParquetFile(right_path)

    if left.metadata.num_rows != right.metadata.num_rows:
        raise AssertionError(f"row count differs: {left.metadata.num_rows} != {right.metadata.num_rows}")
    if not left.schema_arrow.equals(right.schema_arrow):
        raise AssertionError("Parquet schemas differ")

    # same total rows, so zip() cannot silently drop non-empty batches
    for left_batch, right_batch in zip(
        left.iter_batches(batch_size=SHARD_ROWS, use_threads=False),
        right.iter_batches(batch_size=SHARD_ROWS, use_threads=False),
    ):
        pd.testing.assert_frame_equal(left_batch.to_pandas(), right_batch.to_pandas())


def run(files, workers, output_path, shard_dir):
    # process fixed-size shards and merge them into output, returns (n_shards, rows_before, rows_after)
    os.makedirs(shard_dir, exist_ok=True)

    shards = iter_shards(files)
    shards = list(shards) # materialize iterator for progress bar

    if workers == 1: # sequential processing
        results = [process_shard(*shard, shard_dir) for shard in tqdm(shards, desc="Sequential", unit=" shard", leave=False)]
    else: # parallel processing
        results = process_parallel(shards, workers, shard_dir)

    merge_shards([out for out, _, _ in results], output_path)

    return (
        len(results),
        sum(rows_in for _, rows_in, _ in results),
        sum(rows_out for _, _, rows_out in results),
    )


def save_summary(split, shard_rows, workers, threads, sequential, parallel):
    # save summary of run to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = f"{RESULTS_DIR}/summary.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if os.path.getsize(csv_path) == 0:
            writer.writerow(["split", "shard_rows", "workers", "threads", "sequential", "parallel"])
        writer.writerow([split, shard_rows, workers, threads, sequential, parallel])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-rows", type=int, default=SHARD_ROWS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--non-deterministic", action="store_true", default=NON_DETERMINISTIC)
    args = parser.parse_args()

    SHARD_ROWS = args.shard_rows
    WORKERS = args.workers
    THREADS = args.threads
    NON_DETERMINISTIC = args.non_deterministic

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "yellow_tripdata_*.parquet")))

    # time-based split
    splits = {
        "train": [p for p in files if month_of(p) < TEST_START],
        "test": [p for p in files if TEST_START <= month_of(p) <= TEST_END]
    }

    # run the preprocessing tasks in a temporary directory, then move the final Parquet files to PROCESSED_DIR
    with tempfile.TemporaryDirectory(prefix="preprocess_", dir=PROCESSED_DIR) as tmp:
        for split, paths in splits.items():
            started = perf_counter()  # start timer for this split

            print(f"{split} split: {month_of(paths[0])} to {month_of(paths[-1])}")

            # Run once sequentially and once in parallel
            output, seconds = {}, {}

            for mode, workers in (("seq", 1), ("par", WORKERS)):
                output[mode] = os.path.join(tmp, f"{split}_{mode}.parquet")
                mode_started = perf_counter()
                stats = run(paths, workers, output[mode], os.path.join(tmp, f"{split}_{mode}"))
                seconds[mode] = perf_counter() - mode_started

            n_shards, rows_before, rows_after = stats

            print(f"  {len(paths)} files, {n_shards} shards")
            print(f"  {rows_before:,} -> {rows_after:,} trips")
            print(f"  1 workers, {THREADS} threads = {seconds['seq']:.1f} s")
            print(f"  {WORKERS} workers, {THREADS} threads = {seconds['par']:.1f} s")
            print(f"  speedup: {seconds['seq'] / seconds['par']:.1f}x")

            # check sequential and parallel outputs are identical
            assert_parquet_equal(output["seq"], output["par"])
            print("  verified outputs match")

            # move final Parquet file to PROCESSED_DIR
            os.replace(output["par"], os.path.join(PROCESSED_DIR, f"{split}.parquet"))
            print(f"  saved {split}.parquet in {perf_counter() - started:.1f} s")

            # save summary of run
            save_summary(split, SHARD_ROWS, WORKERS, THREADS, seconds["seq"], seconds["par"])
