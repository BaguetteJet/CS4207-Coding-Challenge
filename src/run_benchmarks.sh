#!/usr/bin/env bash
# run_benchmarks.sh - run every sequential vs parallel benchmark on one machine and keep the evidence
#
#   bash run_benchmarks.sh                    # workers 1 2 4, 3 repeats
#   WORKERS="1 2 3" bash run_benchmarks.sh    # e.g. stay within a shared-server limit
#   SKIP_PREPROCESS=1 bash run_benchmarks.sh  # data/processed/ already built on this machine
#
# Everything printed is saved to results/logs/<host>-<date>/, together with the hardware,
# Python version and installed packages, so every number can be traced to where it was run.

set -euo pipefail

WORKERS="${WORKERS:-1 2 4}"
REPEATS="${REPEATS:-3}"
MAX_W="${WORKERS##* }"                       # largest worker count, used for the single-count runs
RUN_DIR="results/logs/$(hostname -s)-$(date +%Y%m%d-%H%M)"
mkdir -p "$RUN_DIR"

# clear result files from earlier runs so only this runs files are copied into RUN_DIR at the end
# (earlier runs are kept in their own results/logs/ folders)
rm -f results/search_*.csv results/gradient_*.csv results/gradient_weights.npz

if ! ls data/yellow_tripdata_*.parquet >/dev/null 2>&1; then
    echo "No data/yellow_tripdata_*.parquet files found. Link or download the dataset first." >&2
    exit 1
fi

# record where these results came from
{
    echo "host: $(hostname)"
    echo "date: $(date)"
    echo "git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    python --version
    echo "cores (logical): $(getconf _NPROCESSORS_ONLN)"
    if command -v lscpu >/dev/null; then
        lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)"
    else
        sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    fi
    echo "data files: $(ls data/yellow_tripdata_*.parquet | wc -l)"
    echo "workers: $WORKERS | repeats: $REPEATS"
} | tee "$RUN_DIR/environment.txt"
pip freeze > "$RUN_DIR/requirements-frozen.txt"

run() {   # run <log name> <command...>: show output live and save it
    local name="$1"; shift
    echo; echo ">>> $*"
    "$@" 2>&1 | tee "$RUN_DIR/$name.log"
}

if [ -z "${SKIP_PREPROCESS:-}" ]; then
    run preprocess python -u src/preprocess.py --workers "$MAX_W"
fi

run search_grid          python -u src/search.py --workers $WORKERS --repeats "$REPEATS"
run search_longest_first python -u src/search.py --workers "$MAX_W" --order longest_first --repeats "$REPEATS"

run gradient_baseline    python -u src/gradient_compute.py --workers 1 --baseline
run gradient_ordered     python -u src/gradient_compute.py --workers $WORKERS --repeats "$REPEATS"
run gradient_completion  python -u src/gradient_compute.py --workers "$MAX_W" --combine completion --repeats "$REPEATS"

# keep this machine's result files together with the logs
cp results/search_*.csv results/gradient_*.csv results/gradient_weights.npz "$RUN_DIR/" 2>/dev/null || true
echo; echo "All done. Logs and results saved in $RUN_DIR"
