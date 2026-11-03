import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

MODEL_PATH = "./models/regression.joblib"

def load(path):
    # Input: trip data
    df = pd.read_parquet(path, columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance"])

    # Target: trip duration in minutes
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

    # Drop obvious bad records
    df = df[df["duration_min"].between(1, 120) & df["trip_distance"].between(0.1, 50)]

    # Features: distance + time of pickup
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["weekday"] = df["tpep_pickup_datetime"].dt.day_of_week
    return df[["trip_distance", "hour", "weekday"]], df["duration_min"]

# Split data
X_train, y_train = load("./data/yellow_tripdata_2026-01.parquet")
X_test, y_test = load("./data/yellow_tripdata_2026-02.parquet")

# Fit the model
model = LinearRegression().fit(X_train, y_train)
joblib.dump(model, MODEL_PATH) # save model

# Evaluate performance
pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, pred):.2f} minutes")
print(f"RMSE: {root_mean_squared_error(y_test, pred):.2f} minutes")

# Sample predictions
print(pd.DataFrame({"actual_min": y_test.values[:5], "predicted_min": pred[:5].round(1)}))