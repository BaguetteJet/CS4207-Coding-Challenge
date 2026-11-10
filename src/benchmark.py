# Benchmark scripts with different configurations

import subprocess
import sys
import os

RESULTS_DIR = "./results/"

def benchmark_preprocess(configs, attempts=1):
    # run the preprocess.py with different configurations and log the outputs
    script = "./src/preprocess.py"
    output_dir = f"{RESULTS_DIR}/preprocess"
    os.makedirs(output_dir, exist_ok=True)

    run = 1
    for shard_rows, workers, threads in configs:
        for _ in range(attempts):
            with open(f"{output_dir}/run{run:02d}-{shard_rows}-{workers}-{threads}.log", "w") as log:
                subprocess.run(
                    [sys.executable, script,
                    "--shard-rows", str(shard_rows),
                    "--workers", str(workers),
                    "--threads", str(threads)],
                    stdout=log,
                    stderr=subprocess.STDOUT
                )
            print(f"Finished run {run}: {shard_rows} shard rows, {workers} workers, {threads} threads")
            run += 1

if __name__ == "__main__":

    # shard_rows, workers, threads
    preprocess_configs = [
        (500_000, 2, 4),
        (500_000, 4, 4),
        (500_000, 8, 4)
    ]
    benchmark_preprocess(preprocess_configs, attempts=1)