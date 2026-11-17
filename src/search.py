# Tudor Ventuneac - 23390131
#hyperparameter search for task parallelism
#evey (model,hyperparemeters) setting in GRID is 1 trial
#trials are independent so same training data, but different config 
# so nothings shared while they run so each trial runs as its own process

# best setting is chosen on the 2025 validation sample


import os

#one maths thread per process set before before numpy or sklearn are imported
#otherwise N processes x M library threads wud fight over the same cores
#could lead to a non deterministic outcome

#added veclib max threads since im on macos and it covers accelerate BLAS used by numpy for macos, should be safe for windows too
for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]= "1"

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import joblib
import numpy
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from preprocess import FEATURES, RATE_CODES, ZONES

#Config
PROCESSED_DIR = "./data/processed"
SEARCH_DIR = "./data/processed/search"
MODEL_DIR= "./models"
RESULTS_DIR = "./results"

SEED = 0
TRAIN_FRAC = 0.02 # using 2% of 2024 rows for the search
VAL_FRAC = 0.02 # 2% of 2025 rows used to score each trial
WORKERS = 4 #we'll change this when running on the rockhill servers to 3
REPEATS= 1 #timed runs per worker count since rubric mentions wanting the mean of 3+ for the benchmark
TARGET = "duration_min"
CATEGORICAL = ["PULocationID","DOLocationID","RatecodeID"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]

#2 ridge trials(alpha barelly mattered in the first search) and 6 gb trials
#first search had its best result on the edge of the grid(highest lr,no depth limit)
# so the grid now goes past it: higher learning rates and more boosting iterations
GRID = (
    [("ridge",{"alpha": a}) for a in (0.1, 10.0)]
    + [("hgb", {"learning_rate": lr,"max_iter":it, "max_depth":None})
       for lr in (0.3,0.5,0.7) for it in (200,500)]
)

def expected_cost(name,params):
    #rough relative runn time of a trial which we'll use to submit the slowest one first
    #a 500 iter hgb trial takes about 2.5 times a 200 iter one, ridgei s the quickest
    return params.get("max_iter",0) if name == "hgb" else 0
 


def make_model(name,params):
    #building an unfitted pipeline for 1 trial
    if name == "ridge":
        #linear model so the zones and rate codes have to be one-hot
        categories = [[float(z) for z in ZONES]] * 2 +[[float(r) for r in RATE_CODES]]
        prep = ColumnTransformer([
            ("onehot",OneHotEncoder(categories=categories, handle_unknown="ignore"), CATEGORICAL),
            ("scale", StandardScaler(), NUMERIC),
        ])
        return Pipeline([("prep",prep), ("model",Ridge(**params))])

    #trees can split on the zone id ranges so dont need one-hot
    #fixed iterations and no early stopping so no random val split inside
    if name == "hgb":
        return Pipeline([("model",HistGradientBoostingRegressor(**params, early_stopping=False,random_state=SEED))])

    raise ValueError(f"unknown model {name}")


#writing a seeded random sample of one years rows of train.parquet to search_dir once and returning its path
def prepare_sample(year,frac):
    os.makedirs(SEARCH_DIR,exist_ok=True)
    out = os.path.join(SEARCH_DIR, f"{year}_{frac}.parquet")
    if os.path.exists(out):
        return out

    parts=[]
    for batch in pq.ParquetFile(os.path.join(PROCESSED_DIR,"train.parquet")).iter_batches(
        batch_size=500_000, columns=FEATURES + [TARGET, "year"], use_threads=False):

        batch = batch.to_pandas()
        year_rows = batch[batch["year"] == year]
        parts.append(year_rows.drop(columns="year").sample(frac=frac,random_state=SEED).astype("float32"))

    pd.concat(parts,ignore_index=True).to_parquet(out,index=False)
    return out



#one search task where itll load the data,fit it,score on val and run inside a worker process
#to make sure main process never pickles the worker will read the data from disk 
def run_trial(trial_id,name,params,train_path,val_path):
    started = perf_counter()

    train = pd.read_parquet(train_path)
    val= pd.read_parquet(val_path)

    model=make_model(name,params).fit(train[FEATURES],train[TARGET])
    pred = model.predict(val[FEATURES])

    return {
        "trial": trial_id,
        "model": name,
        "params": json.dumps(params),
        "val_mae": mean_absolute_error(val[TARGET],pred),
        "val_rmse": root_mean_squared_error(val[TARGET],pred),
        "seconds": perf_counter()-started,
        "pid": os.getpid(),
    }


def search(workers,order,train_path,val_path):
    #running every trial in grid
    tasks = list(enumerate(GRID))
    if order == "longest_first":
        #submiting the most expensive trials first so no worker is left with a long trial at the end
        tasks.sort(key=lambda task: expected_cost(*task[1]), reverse=True)

    ids = [trial_id for trial_id,_ in tasks]
    names = [name for _,(name,_) in tasks]
    params = [p for _, (_,p) in tasks]
    n= len(tasks)


    started = perf_counter()
    #sequential baseline
    if workers == 1:
        rows = [run_trial(*args) for args in zip(ids,names,params, [train_path]*n, [val_path] * n)]
    else: #task parallel->ech trial in its own process and the map returns results in submission order
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_trial,ids,names,params, [train_path] * n, [val_path] * n))
    wall = perf_counter() - started

     #sorting by trial id so table is the same nomatter the order or worker count
    return pd.DataFrame(rows).sort_values("trial").reset_index(drop=True),wall





def main():
    parser = argparse.ArgumentParser(description="parallel hyperparameter search")
    parser.add_argument("--workers",type=int,nargs="+",default=[WORKERS])
    parser.add_argument("--order",choices=["grid","longest_first"],default="grid")
    parser.add_argument("--repeats",type=int,default=REPEATS,help = "timed runs per worker count")
    args = parser.parse_args()

    train_path = prepare_sample(2024,TRAIN_FRAC)
    val_path =prepare_sample(2025,VAL_FRAC)
    print(f"Train sample: {pq.ParquetFile(train_path).metadata.num_rows:,} rows | "
          f"Validation sample: {pq.ParquetFile(val_path).metadata.num_rows:,} rows")
    print(f"{len(GRID)} trials, order = {args.order},{args.repeats} repeats per worker count\n")

    #warming up via reading both samples once untimed so the first timed run doesnt pay for
    #loadingg thme from the disk while the later runs gets them from the OS file cache
    pd.read_parquet(train_path), pd.read_parquet(val_path)

    os.makedirs(RESULTS_DIR,exist_ok=True)
    metrics = ["trial","model","params","val_mae","val_rmse"]
    reference = None   #scores from the very first run, every later run must match it
    tables, runs = {}, []

    for workers in args.workers:
        per_repeat = []
        for repeat in range(1, args.repeats + 1):
            table,wall = search(workers, args.order,train_path, val_path)

            #regardles of worker count or repeat, same trials have to give the same scores
            if reference is None:
                reference = table[metrics]
            pd.testing.assert_frame_equal(reference, table[metrics])

            #summing all the seconds for total work done by all trials
            busy = table["seconds"].sum()
            runs.append({"workers": workers, "order": args.order, "repeat": repeat,
                         "wall_s": wall, "busy_s": busy})
            per_repeat.append(table)
            print(f"  {workers} workers, run {repeat}: wall clock {wall:.1f} s | "
                  f"sum of trial times {busy:.1f} s")

        #per trial times averaged over the repeats (scores are identical so they're just kept)
        mean_table = per_repeat[0].copy()
        mean_table["seconds"] = sum(t["seconds"] for t in per_repeat) / len(per_repeat)
        tables[workers] = mean_table

        print(f">>> {workers} workers (trial seconds = mean of {args.repeats} run(s))")
        print(mean_table.drop(columns="pid").round(3).to_string(index=False))
        print()
        mean_table.drop(columns="pid").to_csv(
            os.path.join(RESULTS_DIR, f"search_{workers}w_{args.order}.csv"),index=False)

    print("verified: identical validation scores for", ", ".join(map(str,args.workers)),
          f"workers across {args.repeats} repeat(s)")

    #every timed run,+ a summary with mean and spread per worker count
    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(RESULTS_DIR, f"search_runs_{args.order}.csv"), index=False)

    summary = runs.groupby("workers").agg(
        wall_mean=("wall_s","mean"), wall_std=("wall_s","std"),
        busy_mean=("busy_s","mean"), runs=("repeat","count")).reset_index()
    if 1 in args.workers:
        base = summary.loc[summary["workers"] == 1, "wall_mean"].iloc[0]
        summary["speedup"] = base / summary["wall_mean"]
        summary["efficiency"] = summary["speedup"] / summary["workers"]
    summary.to_csv(os.path.join(RESULTS_DIR, f"search_timings_{args.order}.csv"), index=False)
    print("\n" + summary.round(3).to_string(index=False))

 
    # pick the best setting on val rmse, refit it, save it for evaluate.py
    best = reference.loc[reference["val_rmse"].idxmin()]
    print(f"\nBest: {best['model']} {best['params']} | val RMSE {best['val_rmse']:.3f} | val MAE {best['val_mae']:.3f}")
 
    train = pd.read_parquet(train_path)
    model = make_model(best["model"], json.loads(best["params"])).fit(train[FEATURES], train[TARGET])
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "search_best.joblib"))
    print("Saved models/search_best.joblib (scored on test by evaluate.py)")
 
 
if __name__ == "__main__":
    main()