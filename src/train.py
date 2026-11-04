import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from preprocess import FEATURES

TRAIN_PATH = "./data/processed/train.parquet"
MODEL_DIR = "./models"

MODELS = {
    "linear_regression": LinearRegression(),
    # add more models here
}

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_parquet(TRAIN_PATH)
    X_train, y_train = df[FEATURES], df["duration_min"]

    # Fit models with preprocessing pipeline
    for name, estimator in MODELS.items():
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]).fit(X_train, y_train)

        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"Trained {name}")
