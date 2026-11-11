# Guide

This is a quick guide to get started on the project.

## STEP 1 - Setup Environment

Requires Python 3.10+

Create virtual environment and activate it
```bash
# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

Install requirements
```bash
pip install -r requirements.txt
```

For development, setup the local Git configuration to remove `.ipynb` notebook outputs from commits
```bash
nbstripout --install
```

Also please use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) to keep commit history tidy

## STEP 2 - Prepare Dataset

Run [`download.py`](./src/download.py) to automatically download dataset files into `data/`

This project uses data from 2024-01 to 2026-09

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## STEP 3 - Preprocess

Run [`preprocess.py`](./src/preprocess.py) to extract features

Parquet files are read from `data/` and sharded into equal smaller chunks, which are processed in sequence and in parallel. Processing includes data cleaning and feature extraction. Results of both are verified to be identical.

- `SHARD_ROWS` - number of rows per shard
- `WORKERS` - number of processes for sharding (1=sequential, N=parallel)
- `THREADS` - number of threads for feature extraction (1=sequential, N=parallel)

Progress is done in batches and saved to a temporary directory before merging to reduce memory usage.

Train/Test splits saved to `data/processed/` as `.parquet` files

## Train

Run [`train.py`](./src/train.py) to fit the models

Models with the preprocessing pipeline included are saved to `models/` as a `.joblib` files

## Evaluate

Run [`evaluate.py`](./src/evaluate.py) to compare saved models on test split

MAE, RMSE and R^2 metrics saved to `results/metrics.csv`

## Prediction

Run [`predict.py`](./src/predict.py) with a saved model to predict the duration of new trips

```bash
python src/predict.py models/random_forest.joblib
python src/predict.py models/random_forest.joblib --input trips.csv
```

Input need at least `tpep_pickup_datetime`, `PULocationID` and `DOLocationID` columns

## Rockhill Server

> See [server documentation](/docs/SERVER.md) for setup and details.

Log into lab computer using you student email and password.

SSH into the server using the group login.

```bash
ssh group19@rockhill.csis.ul.ie
```

Run [`rockhill.csis.ul.ie.bat`](/rockhill.csis.ul.ie.bat) from within `/CS4207-Coding-Challenge/` on Windows lab computer to upload/download files between lab computer and server.
