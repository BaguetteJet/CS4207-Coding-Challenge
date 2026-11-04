import joblib
import pandas as pd
import argparse

from preprocess import FEATURES, extract_features

parser = argparse.ArgumentParser(description="Predict NYC taxi trip duration")
parser.add_argument("model", nargs="?", default="./models/linear_regression.joblib", help="Path to saved model")
parser.add_argument("--input", help="CSV or Parquet file with required columns")
args = parser.parse_args()

model = joblib.load(args.model)

if args.input:
    read = pd.read_parquet if args.input.endswith(".parquet") else pd.read_csv
    trips = read(args.input)
else:
    # New trip(s)
    trips = pd.DataFrame([
    {
        "VendorID": 2,
        "tpep_pickup_datetime": "2026-04-06 08:15:00",
        "tpep_dropoff_datetime": None,
        "passenger_count": 1,
        "trip_distance": None,
        "RatecodeID": 1,
        "store_and_fwd_flag": "N",
        "PULocationID": 161,
        "DOLocationID": 236,
        "payment_type": None,
        "fare_amount": None,
        "extra": None,
        "mta_tax": None,
        "tip_amount": None,
        "tolls_amount": None,
        "improvement_surcharge": None,
        "total_amount": None,
        "congestion_surcharge": None,
        "airport_fee": None,
        "cbd_congestion_fee": None
    }
])

# Build features expected by model
features = getattr(model, "feature_names_in_", FEATURES)
trips["eta_min"] = model.predict(extract_features(trips)[features]).round(1)
print(trips)
