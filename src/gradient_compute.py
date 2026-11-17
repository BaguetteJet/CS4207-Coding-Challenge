#Tudor Ventuneac - 23390131
#linear regression for trip duration,trained by batch gradient descent
#gonna break it down in terms of broadcasting, compting, combining and then updating
#1 broadcast, current weights are sent to every shard task
#2 compute, each task will return the sums of its trips gradients plus the trip count
#not avg since the shards have diff sizes
#3 combine, main process where itll add the sum in shard order and divde by total count
#4 update, 1 gradient step on the single shared model

import os
for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"

import argparse
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import numpy as np
import pandas as pd
from preprocess import FEATURES,RATE_CODES
from search import CATEGORICAL,SEED,TARGET, make_model, prepare_sample


#config
PREP_DIR= "./data/processed/gradient"
RESULTS_DIR = "./results"

TRAIN_FRAC=0.20 #20% of the 2024 trips, should be around 8mill so each step has enough work to parallelise
VAL_FRAC= 0.02 #same 2025 val sample you changed in search
SHARD_ROWS = 1000000 #1 mill shards per row, equal sized, last one gonna be smaller

ETA=0.2 #lr
LAM=0.0 #l2 reg. strength, not applied to the intercept
STEPS=200 #fixing the number of steps so every run does the exact same amount of work
WORKERS = 4
REPEATS = 1

NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
N_ZONES= 266 #zone ids 1-265
N_RATES = len(RATE_CODES)


def to_arrays(df):
    #storing the one hot cols as small int codes instead mostly zero cols
    return{
        "num": df[NUMERIC].to_numpy(np.float32),
        "pu": df["PULocationID"].to_numpy().astype(np.int16),
        "do": df["DOLocationID"].to_numpy(np.int16),
        "rate":np.searchsorted(RATE_CODES, df["RatecodeID"].to_numpy()).astype(np.int8),
        "y":df[TARGET].to_numpy(np.float32),
    }

def save_arrays(prefix,arrays):
    for name,values in arrays.items():
        np.save(f"{prefix}_{name}.npy", values)


# memory mapped so the os reads pages on demand an shares them between processes
def load_arrays(prefix):
    return {name:np.load(f"{prefix}_{name}.npy",mmap_mode="r")
            for name in ("num","pu","do","rate","y")}


#sampling once and cutting the training sample into fixed shards and saving val, itll be reused on later runs
def prepare_shards():
    meta_path = os.path.join(PREP_DIR, f"meta_{TRAIN_FRAC}_{SHARD_ROWS}.json")
    if os.path.exists(meta_path):
        return json.load(open(meta_path))

    os.makedirs(PREP_DIR,exist_ok=True)

    train = pd.read_parquet(prepare_sample(2024,TRAIN_FRAC))
    val = pd.read_parquet(prepare_sample(2025,VAL_FRAC))

    rows = []
    for shard_id,start in enumerate(range(0,len(train), SHARD_ROWS)):
        shard = train.iloc[start:start+SHARD_ROWS]
        save_arrays(os.path.join(PREP_DIR,f"s{shard_id:02d}"), to_arrays(shard))
        rows.append(len(shard))
    save_arrays(os.path.join(PREP_DIR,"val"),to_arrays(val))

    meta= {"shard_rows":rows,"train_rows":len(train),"val_rows":len(val)}
    json.dump(meta,open(meta_path,"w"))
    return meta


def init_params():
    #same starting point every run with all weights 0
    return {"num":np.zeros(len(NUMERIC)),"pu":np.zeros(N_ZONES),"do":np.zeros(N_ZONES), "rate":np.zeros(N_RATES),"b":0.0}

#b + numeric weights + one weight per pickup zone,dropoff zone and rate code
def predict(p,d):
    return p["b"] + d["num"] @ p["num"] + p["pu"][d["pu"]] + p["do"][d["do"]] + p["rate"][d["rate"]]


#one task which is summed gradients for 1 shard using the broadcast weights p
#opens its shard from disk via mmap so nothing large get pickled, it calculates each trips error with p nad returns sums plus count
def shard_gradient(shard_id,prep_dir,p):
    d= load_arrays(os.path.join(prep_dir,f"s{shard_id:02d}"))
    #error per trip
    err = predict(p,d) - d["y"]

    return{
        "num": d["num"].T @ err, #sum of error times input
        "pu": np.bincount(d["pu"],weights=err, minlength=N_ZONES), #onehot-> sum of errors per zone
        "do": np.bincount(d["do"], weights=err, minlength=N_ZONES),
        "rate": np.bincount(d["rate"], weights=err, minlength=N_RATES),
        "b": err.sum(), #intercept(sum of errors)
        "sse": err @ err, #for the training rmse
        "n": len(err) #count so we divide once at the end
    }

#adding the shards sum in the order given
def combine(parts):
    #adding the partial sums in the order given(fp addition depends on the order)
    return {key:sum(part[key] for part in parts)for key in parts[0]}

#training loop
#broadcast->compute->combine->divide by total count->1 update
def run_gd(workers,combine_mode,n_shards,steps=STEPS,eta=ETA,lam=LAM):
    #bgd, workers=1 runs the shards one after another in this process
    p,history = init_params(),[]
    ids = list(range(n_shards))
    pool = ProcessPoolExecutor(max_workers=workers) if workers>1 else None

    started = perf_counter()
    try:
        for _ in range(steps):
            if pool is None: #sequential baseline
                parts = [shard_gradient(i,PREP_DIR,p)for i in ids]
            elif combine_mode == "ordered": #map returns results in shared order
                parts = list(pool.map(shard_gradient, ids, [PREP_DIR] * n_shards, [p] * n_shards))
            else:
                futures = [pool.submit(shard_gradient, i, PREP_DIR, p) for i in ids]
                parts = [f.result() for f in as_completed(futures)]

            G= combine(parts)
            N = G["n"]
            for key in ("num","pu","do","rate"):
                p[key] = p[key] - eta * (G[key] / N + lam * p[key])
            p["b"] = p["b"] - eta * G["b"] / N
            #training rmse at the start of this step
            history.append(np.sqrt(G["sse"]/N))

    finally:
        if pool:
            pool.shutdown()
    return p,history,perf_counter() - started

def flat(p):
    return np.concatenate([p["num"],p["pu"],p["do"],p["rate"],[p["b"]]])


def score(p,d):
    err = predict(p,d)-d["y"]
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))



def main():
    parser = argparse.ArgumentParser(description="parallel gradient computation")
    parser.add_argument("--workers", type=int, nargs="+", default=[WORKERS])
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--combine", choices=["ordered", "completion"], default="ordered")
    parser.add_argument("--baseline", action="store_true", help="also fit scikitlearn ridge")
    args = parser.parse_args()

    meta = prepare_shards()
    n_shards = len(meta["shard_rows"])
    print(f"Train: {meta['train_rows']:,} trips in {n_shards} shards {meta['shard_rows']} | "
          f"Validation: {meta['val_rows']:,} trips")
    print(f"{args.steps} steps, eta = {ETA}, lambda = {LAM}, combine = {args.combine}, "
          f"{args.repeats} repeat(s)\n")


    #warmup,one untimed pass so the os has the shard files cached before timing starts
    for i in range(n_shards):
        shard_gradient(i, PREP_DIR, init_params())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    reference, runs = None, []



    for workers in args.workers:
        for repeat in range(1, args.repeats + 1):
            p, history, wall = run_gd(workers, args.combine, n_shards,steps=args.steps)
            weights = flat(p)

            if reference is None:
                reference, best_p, best_history = weights, p, history
            identical = bool(np.array_equal(reference, weights))
            max_diff = float(np.max(np.abs(reference - weights)))

            runs.append({"workers": workers, "combine": args.combine, "repeat": repeat,
                         "wall_s": wall, "step_ms": 1000 * wall /args.steps,
                         "identical": identical, "max_abs_diff": max_diff})
            
            print(f"  {workers} workers, run {repeat}: {wall:.1f} s ({1000 * wall / args.steps:.0f} ms/step) | "
                  f"final train RMSE {history[-1]:.4f} | identical to first run: {identical} "
                  f"(max diff {max_diff:.2e})")


    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(RESULTS_DIR, f"gradient_runs_{args.combine}.csv"), index=False)

    summary = runs.groupby("workers").agg(
        wall_mean=("wall_s", "mean"), wall_std=("wall_s", "std"),
        step_ms=("step_ms", "mean"), runs=("repeat", "count"),
        all_identical=("identical", "all")).reset_index()

    if 1 in args.workers:
        base = summary.loc[summary["workers"] == 1, "wall_mean"].iloc[0]
        summary["speedup"] = base / summary["wall_mean"]
        summary["efficiency"] = summary["speedup"] / summary["workers"]
    summary.to_csv(os.path.join(RESULTS_DIR, f"gradient_timings_{args.combine}.csv"), index=False)
    print("\n" + summary.round(3).to_string(index=False))

    if runs["identical"].all():
        print("\nverified: bit identical weights for", ", ".join(map(str, args.workers)),
              f"workers across {args.repeats} repeats")
    else:
        print(f"\nNOT identical: weights differ by up to {runs['max_abs_diff'].max():.2e} between runs")


    #convergence curve for the poster and val score of trained model
    pd.DataFrame({"step": range(1, len(best_history) + 1), "train_rmse": best_history}).to_csv(
        os.path.join(RESULTS_DIR, "gradient_history.csv"), index=False)
    val_rmse, val_mae = score(best_p, load_arrays(os.path.join(PREP_DIR, "val")))
    print(f"\nGradient descent: val RMSE {val_rmse:.3f} | val MAE {val_mae:.3f}")
    np.savez(os.path.join(RESULTS_DIR, "gradient_weights.npz"), **{k: np.atleast_1d(v) for k, v in best_p.items()})

    if args.baseline:
        #closed form style solver on the same training trips for comparison
        train = pd.read_parquet(prepare_sample(2024, TRAIN_FRAC))
        val = pd.read_parquet(prepare_sample(2025, VAL_FRAC))
        started = perf_counter()
        ridge = make_model("ridge", {"alpha": 1.0}).fit(train[FEATURES], train[TARGET])
        seconds = perf_counter() - started
        pred = ridge.predict(val[FEATURES])
        rmse = float(np.sqrt(np.mean((pred - val[TARGET]) ** 2)))
        mae = float(np.mean(np.abs(pred - val[TARGET])))
        print(f"scikitlearn Ridge: val RMSE {rmse:.3f} | val MAE {mae:.3f} | fit {seconds:.1f} s")

if __name__=="__main__":
    main()