#hyperparameter search for task parallelism
#evey (model,hyperparemeters) setting in GRID is 1 trial
#trials are independent so same training data, but different config 
# so nothings shared while they run so each trial runs as its own process

# best setting is chosen on the val split(july-dec 2025)


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
TRAIN_FRAC = 0.02 # using 2% of training rows for the search(around 1.2mill out of the 61mill)
VAL_FRAC = 0.05 #5% of val rows used to score each trial(around 1.1m of 22mill)
WORKERS = 4 #we'll change this when running on the rockhill servers to 3

TARGET = "duration_min"
CATEGORICAL = ["PULocationID","DOLocationID","RatecodeID"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]

#4 ridge regression trials and 4 gb trials
GRID = (
    [("ridge",{"alpha": a}) for a in (0.1, 1.0, 10.0, 100.0)]
    + [("hgb", {"learning_rate": lr, "max_depth": d})
       for lr in (0.1,0.3) for d in (6,None)]
)




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
        return Pipeline([("model",HistGradientBoostingRegressor(**params,max_iter = 200, early_stopping=False,random_state=SEED))])

    raise ValueError(f"unknown model {name}")


#writing a seeded random sample of one split to search_dir once and returning its path
def prepare_sample(split,frac):
    os.makedirs(SEARCH_DIR,exist_ok=True)
    out = os.path.join(SEARCH_DIR, f"{split}_{frac}.parquet")
    if os.path.exists(out):
        return out

    parts=[]
    for batch in pq.ParquetFile(os.path.join(PROCESSED_DIR,f"{split}.parquet")).iter_batches(
        batch_size=500_000, columns=FEATURES + [TARGET], use_threads= False):

        parts.append(batch.to_pandas().sample(frac=frac,random_state=SEED).astype("float32"))

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
        #starting the gb trials first since theyre slower than ridge
        tasks.sort(key=lambda task:task[1][0] != "hgb")

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
    args = parser.parse_args()

    train_path = prepare_sample("train",TRAIN_FRAC)
    val_path =prepare_sample("val",VAL_FRAC)
    print(f"Train sample: {pq.ParquetFile(train_path).metadata.num_rows:,} rows |"
          f"Validation sample: {pq.ParquetFile(val_path).metadata.num_rows:,}rows")
    print(f"{len(GRID)} trials, order = {args.order}\n")

    os.makedirs(RESULTS_DIR,exist_ok=True)
    tables,timings={},[]

    for workers in args.workers:
        table,wall = search(workers, args.order,train_path, val_path)
        tables[workers]= table
        #summing all the seconds for total work done by all triials
        busy = table["seconds"].sum()
        print(f">>> {workers} workers")


        print(table.drop(columns="pid").round(3).to_string(index=False))
        print(f"wall clock {wall:.1f} s | sum of trial times {busy:.1f} s | "
              f"longest trial {table['seconds'].max():.1f} s\n")

        timings.append({"workers": workers, "order": args.order, "wall_s": wall, "busy_s": busy})
        table.to_csv(os.path.join(RESULTS_DIR, f"search_{workers}w_{args.order}.csv"),index=False)


#regardles of worker count same trials haveto give he same scores
    metrics = ["trial","model","params","val_mae","val_rmse"]
    first = args.workers[0]
    for workers in args.workers[1:]:
        pd.testing.assert_frame_equal(tables[first][metrics], tables[workers][metrics])

    if len(args.workers) >1:
        print("verified:identical validation scores for", ", ".join(map(str,args.workers)),"workers")
        base = next((t["wall_s"] for t in timings if t["workers"] == 1),None)
        if base:
            for t in timings:
                speedup = base/t["wall_s"]
                print(f" {t['workers']} workers: {t['wall_s']:.1f} s, speedup {speedup:.2f}x, "
                        f"efficiency {speedup/t['workers']:.2f}")


    pd.DataFrame(timings).to_csv(os.path.join(RESULTS_DIR, f"search_timings_{args.order}.csv"), index=False)
 
    # pick the best setting on val rmse, refit it, save it for evaluate.py
    best = tables[first].loc[tables[first]["val_rmse"].idxmin()]
    print(f"\nBest: {best['model']} {best['params']} | val RMSE {best['val_rmse']:.3f} | val MAE {best['val_mae']:.3f}")
 
    train = pd.read_parquet(train_path)
    model = make_model(best["model"], json.loads(best["params"])).fit(train[FEATURES], train[TARGET])
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "search_best.joblib"))
    print("Saved models/search_best.joblib (scored on test by evaluate.py)")
 
 
if __name__ == "__main__":
    main()