import os
import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from preprocess import FEATURES

TRAIN_PATH = "./data/processed/train.parquet"
MODEL_DIR = "./models"
SAMPLE_FRAC = 1.0  # fraction of rows to train on; raise it until memory gets tight

MODELS = {
    "linear_regression": LinearRegression(copy_X=False),  # no extra copy
    # add more models here
}

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    # read in batches and keep a sample, so the full table is never in memory
    parts = []
    for batch in pq.ParquetFile(TRAIN_PATH).iter_batches(batch_size=500_000, columns=FEATURES + ["duration_min"]):
        parts.append(batch.to_pandas().sample(frac=SAMPLE_FRAC, random_state=0).astype("float32"))

    df = pd.concat(parts, ignore_index=True)
    del parts

    X_train = df[FEATURES]
    y_train = df["duration_min"]
    del df

    # Fit models with preprocessing pipeline
    for name, estimator in MODELS.items():
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]).fit(X_train, y_train)

        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"Trained {name}")