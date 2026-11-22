import os
import argparse
import json
from time import perf_counter
import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from preprocess import FEATURES

TRAIN_PATH = "./data/processed/train.parquet"
MODEL_DIR = "./models"
RESULTS_DIR = "./results"
BEST_PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")  # written by tune_models.py
SAMPLE_FRAC = 0.05  # fraction of rows to train on; full data is too big for the tree models
SEED = 0

# fixed seeds so retraining gives the same models
# hyperparameters picked on the search.py samples (train on 2024, score on 2025 val), never on test
MODELS = {
    "linear_regression": LinearRegression(copy_X=False),  # no extra copy
    # defaults below come from tune_models.py (results/model_tuning.csv), results/best_params.json overrides them
    # tried max_depth 10/15/20/None x min_samples_leaf 50/200, deeper with small leaves won (val RMSE 8.43)
    # min_samples_leaf caps the tree size even with no depth limit
    "decision_tree": DecisionTreeRegressor(max_depth=None, min_samples_leaf=50, random_state=SEED),
    # depth 20 beat 15 (val RMSE 8.30 vs 8.62), 100 trees no better than 50 for twice the time and size
    "random_forest": RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=50, n_jobs=-1, random_state=SEED),
    # best of the widened search.py grid (val RMSE 7.36), no early stopping so no random validation split
    "hist_gradient_boosting": HistGradientBoostingRegressor(learning_rate=0.5, max_depth=None, max_iter=500, early_stopping=False, random_state=SEED),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the models and save them to models/")
    parser.add_argument("--jobs", type=int, default=-1, help="random forest processes (-1 = all cores)")
    args = parser.parse_args()

    # use the tuned settings from tune_models.py if it has been run, otherwise the defaults above
    best_model = None
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            best_params = json.load(f)
        best_model = best_params.pop("best_model")
        for name, params in best_params.items():
            MODELS[name].set_params(**params)
        print(f"Using tuned settings from {BEST_PARAMS_PATH}, best model on validation: {best_model}")
    else:
        print(f"No {BEST_PARAMS_PATH}, using default settings (run tune_models.py first to tune)")
    MODELS["random_forest"].set_params(n_jobs=args.jobs)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # read in batches and keep a sample, so the full table is never in memory
    parts = []
    for batch in pq.ParquetFile(TRAIN_PATH).iter_batches(batch_size=500_000, columns=FEATURES + ["duration_min"]):
        parts.append(batch.to_pandas().sample(frac=SAMPLE_FRAC, random_state=SEED).astype("float32"))

    df = pd.concat(parts, ignore_index=True)
    del parts

    X_train = df[FEATURES]
    y_train = df["duration_min"]
    del df

    # Fit models with preprocessing pipeline
    times = []
    for name, estimator in MODELS.items():
        started = perf_counter()
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]).fit(X_train, y_train)
        seconds = perf_counter() - started

        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        if name == best_model:
            # chosen on validation, so evaluate.py's test score for it is a fair estimate
            joblib.dump(model, os.path.join(MODEL_DIR, "best_model.joblib"))
            print(f"Saved {name} as best_model.joblib")
        times.append({"model": name, "train_rows": len(X_train), "train_s": seconds})
        print(f"Trained {name} in {seconds:.1f} s")

    pd.DataFrame(times).to_csv(os.path.join(RESULTS_DIR, "train_times.csv"), index=False)