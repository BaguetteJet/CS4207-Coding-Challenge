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

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Download dataset files:

- [`yellow_tripdata_2026-01.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet) 61.2 MB
- [`yellow_tripdata_2026-02.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-02.parquet) 52.0 MB
- [`yellow_tripdata_2026-03.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-03.parquet) 64.7 MB
- [`yellow_tripdata_2026-04.parquet`](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-04.parquet) 61.8 MB

