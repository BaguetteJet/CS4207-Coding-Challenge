import joblib
import pandas as pd

MODEL_PATH = "./models/regression.joblib"

model = joblib.load(MODEL_PATH)

# New trips
trips = pd.DataFrame({
    "trip_distance": [3.5, 1.2, 12.0],
    "pickup": pd.to_datetime(["2025-05-03 18:00", "2025-05-08 02:30", "2025-05-21 09:15"]),
})

# Build features
X = pd.DataFrame({
    "trip_distance": trips["trip_distance"],
    "hour": trips["pickup"].dt.hour,
    "weekday": trips["pickup"].dt.day_of_week,
})

trips["eta_min"] = model.predict(X).round(1)
print(trips)