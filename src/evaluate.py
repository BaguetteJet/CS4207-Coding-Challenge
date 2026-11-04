import os
import glob
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from preprocess import FEATURES

TEST_PATH = "./data/processed/test.parquet"
RESULTS_DIR = "./results"

if __name__ == "__main__":
    model_paths = sorted(glob.glob("./models/*.joblib"))

    df = pd.read_parquet(TEST_PATH)
    y_test = df["duration_min"]

    # Score saved models on test set
    rows = []
    for path in model_paths:
        model = joblib.load(path)
        features = getattr(model, "feature_names_in_", FEATURES)
        pred = model.predict(df[features])

        rows.append({
            "model": os.path.splitext(os.path.basename(path))[0],
            "mae": mean_absolute_error(y_test, pred),
            "rmse": root_mean_squared_error(y_test, pred),
            "r2": r2_score(y_test, pred),
        })

    results = pd.DataFrame(rows).sort_values("rmse")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

    print(results.round(3).to_string(index=False))
    print(f"Saved results")
