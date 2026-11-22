# hyperparameter tuning for the models in train.py
# same samples as search.py: train on 2024, score on 2025 validation, test is never used
# every setting is an independent trial, run in its own process

import search  # sets one maths thread per process, has to come before numpy/sklearn

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from preprocess import FEATURES

RESULTS_DIR = "./results"
BEST_PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")
SEED = 0

GRID = (
    [("decision_tree", {"max_depth": d, "min_samples_leaf": l}) for d in (10, 15, 20, None) for l in (50, 200)]
    # 100 trees scored the same as 50 (val RMSE 8.293 vs 8.296) for twice the time and size, so only 50 is tried
    + [("random_forest", {"n_estimators": 50, "max_depth": d, "min_samples_leaf": 50}) for d in (15, 20)]
    + [("hist_gradient_boosting", p) for name, p in search.GRID if name == "hgb"]
)


def make_model(name, params):
    if name == "decision_tree":
        return DecisionTreeRegressor(**params, random_state=SEED)
    if name == "random_forest":
        return RandomForestRegressor(**params, n_jobs=1, random_state=SEED)  # the pool gives the parallelism
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**params, early_stopping=False, random_state=SEED)
    raise ValueError(f"unknown model {name}")


def run_trial(name, params, train_path, val_path):
    started = perf_counter()
    train, val = pd.read_parquet(train_path), pd.read_parquet(val_path)
    pred = make_model(name, params).fit(train[FEATURES], train[search.TARGET]).predict(val[FEATURES])
    return {
        "model": name,
        "params": json.dumps(params),
        "val_mae": mean_absolute_error(val[search.TARGET], pred),
        "val_rmse": root_mean_squared_error(val[search.TARGET], pred),
        "seconds": perf_counter() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tune the train.py models on the validation sample")
    parser.add_argument("--workers", type=int, default=search.WORKERS)
    args = parser.parse_args()

    train_path = search.prepare_sample(2024, search.TRAIN_FRAC)
    val_path = search.prepare_sample(2025, search.VAL_FRAC)
    n = len(GRID)
    print(f"{n} trials, {args.workers} workers")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_trial, [g[0] for g in GRID], [g[1] for g in GRID], [train_path] * n, [val_path] * n))

    results = pd.DataFrame(rows).sort_values(["model", "val_rmse"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "model_tuning.csv"), index=False)
    print(results.round(3).to_string(index=False))

    # best setting per model and the best model overall, all on val RMSE, read by train.py
    best_params = {}
    print("\nBest per model:")
    for name, group in results.groupby("model"):
        best = group.iloc[0]
        best_params[name] = json.loads(best["params"])
        print(f"  {name}: {best['params']} | val RMSE {best['val_rmse']:.3f}")
    best_params["best_model"] = results.loc[results["val_rmse"].idxmin(), "model"]
    print(f"Best model: {best_params['best_model']}")

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved {BEST_PARAMS_PATH} (used by train.py)")
