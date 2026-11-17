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

## STEP 4 - Hyperparameter Search

Run [`search.py`](./src/search.py) to tune the models (task parallelism)

Every (model, hyperparameters) setting in `GRID` is an independent trial, run in its own process. Trials train on a seeded sample of 2024 trips and are scored on a sample of 2025 trips (validation), so the 2026 test split is never used for model selection.

```bash
python src/search.py                                                  # one run with 4 workers
python src/search.py --workers 1 2 4 --repeats 3                      # sequential vs parallel benchmark
python src/search.py --workers 4 --order longest_first --repeats 3    # submit the slowest trials first
```

- `--workers` - one or more worker counts (1=sequential, N=parallel)
- `--repeats` - timed runs per worker count, results reported as the mean
- `--order` - `grid` (grid order) or `longest_first` (reduces load imbalance)

Validation scores are verified to be identical for every worker count and repeat. The best setting (lowest validation RMSE) is refit and saved to `models/search_best.joblib`.

Results saved to `results/`: `search_<N>w_<order>.csv` (per-trial scores and times), `search_runs_<order>.csv` (every timed run) and `search_timings_<order>.csv` (mean time, speedup, efficiency)

## STEP 5 - Gradient Computation

Run [`gradient_compute.py`](./src/gradient_compute.py) to train linear regression by batch gradient descent with gradients computed in parallel (task parallelism)

Each step the current weights are sent to every shard, each worker returns the summed gradients and trip count for its shard, and the main process adds them in shard order before one weight update.

```bash
python src/gradient_compute.py --workers 1 --steps 200 --baseline              # one run, compared with scikit-learn Ridge
python src/gradient_compute.py --workers 1 2 4 --repeats 3                     # sequential vs parallel benchmark
python src/gradient_compute.py --workers 4 --combine completion --repeats 3    # non-determinism demo
```

- `--workers` / `--repeats` - as for `search.py`
- `--steps` - number of gradient descent steps
- `--combine` - `ordered` (shard order, reproducible) or `completion` (finish order, shows floating-point non-determinism)
- `--baseline` - also fit scikit-learn Ridge on the same trips

The first run samples 20% of 2024 trips and saves them as 1M-row shards in `data/processed/gradient/` (requires `search.py` in `src/`); later runs reuse them. Final weights are checked to be bit-identical across worker counts and repeats.

Results saved to `results/`: `gradient_runs_<combine>.csv`, `gradient_timings_<combine>.csv`, `gradient_history.csv` (training RMSE per step) and `gradient_weights.npz`

## Train

Run [`train.py`](./src/train.py) to fit the models

Models with the preprocessing pipeline included are saved to `models/` as a `.joblib` files

## Evaluate

Run [`evaluate.py`](./src/evaluate.py) to compare saved models on test split

Includes `search_best.joblib` from the hyperparameter search

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
