# Benchmark scripts with different configurations

import sys
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = "./src/"
RESULTS_DIR = "./results/"

def benchmark_preprocess(configs, attempts=1):
    # run the preprocess.py with different configurations and log the outputs
    script = f"{SRC_DIR}/preprocess.py"
    output_dir = f"{RESULTS_DIR}/preprocess"
    os.makedirs(output_dir, exist_ok=True)

    run = 1
    for shard_rows, workers, threads in configs:
        for _ in range(attempts):
            with open(f"{output_dir}/run{run:03d}-{shard_rows}-{workers}-{threads}.log", "w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", script,
                    "--shard-rows", str(shard_rows),
                    "--workers", str(workers),
                    "--threads", str(threads)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Print and log output while preprocessing runs
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()

                return_code = process.wait()

                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, process.args)

            print(f"Finished run {run:03d}: {shard_rows} shard rows, {workers} workers, {threads} threads")
            run += 1


def plot_preprocess_results():
    df = pd.read_csv(f"{RESULTS_DIR}/preprocess/summary.csv")

    avg = df.groupby(["split", "shard_rows", "workers", "threads"])[["sequential", "parallel"]].mean().reset_index()

    avg["speedup"] = avg["sequential"] / avg["parallel"]

    configs = avg[["shard_rows", "workers", "threads"]].drop_duplicates()
    labels = [f"{r.shard_rows:,}\n{r.workers} workers\n{r.threads} threads" for r in configs.itertuples(index=False)]

    x = np.arange(len(configs))
    width = 0.35

    fig, axes = plt.subplots(1, avg["split"].nunique(), figsize=(7 * avg["split"].nunique(), 6), dpi=180)
    axes = np.atleast_1d(axes)

    for ax, (split, data) in zip(axes, avg.groupby("split")):
        data = data.sort_values(["shard_rows", "workers", "threads"])

        ax.bar(x - width / 2, data["sequential"], width, label="Sequential")
        ax.bar(x + width / 2, data["parallel"], width, label="Parallel")

        for i, row in enumerate(data.itertuples()):
            ax.text(x[i], max(row.sequential, row.parallel) * 1.01, f"{row.speedup:.2f}x", ha="center", va="bottom", fontweight="bold")

        ax.set(title=f"{split.capitalize()} preprocessing", xlabel="Configuration", ylabel="Average execution time (seconds)", xticks=x, xticklabels=labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Average Sequential vs Parallel Preprocessing Time")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/preprocess/benchmark.png",dpi=300, bbox_inches="tight")
    plt.show()

    print("\nAverage execution times and speedups:")
    print(avg.to_string(index=False))


if __name__ == "__main__":

    
    #shard_rows, workers, threads
    preprocess_configs = [
        # Small shards
        (100_000, 1, 1),
        (100_000, 2, 2),
        (100_000, 4, 2),
        (100_000, 8, 2),
        (100_000, 4, 4),
        (100_000, 8, 4),

        # Large shards
        (1_000_000, 1, 1),
        (1_000_000, 2, 2),
        (1_000_000, 4, 2),
        (1_000_000, 8, 2),
        (1_000_000, 4, 4),
        (1_000_000, 8, 4),
    ]
    benchmark_preprocess(preprocess_configs, attempts=3)
    plot_preprocess_results()

