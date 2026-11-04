# Guide

This is a quick guide to get started on the project.

## Setup Environment

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

## Prepare Dataset
Run [`download.py`](./src/download.py) to automatically download dataset files into `data/`

or manually download files into `data/`:

- [`yellow_tripdata_2026-01.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet) 61.2 MB
- [`yellow_tripdata_2026-02.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-02.parquet) 52.0 MB
- [`yellow_tripdata_2026-03.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-03.parquet) 64.7 MB
- [`yellow_tripdata_2026-04.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-04.parquet) 61.8 MB

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Preprocess

Run [`preprocess.py`](./src/preprocess.py) to extract features and save the processed data

Train/Test split saved to `data/processed/` as `.parquet` files (Jan-Mar train, Apr test)

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

