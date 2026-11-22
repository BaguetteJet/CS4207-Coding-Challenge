#!/usr/bin/env bash
# run_models.sh - tune, train and evaluate the regression models on one machine and keep the evidence
#
#   bash src/run_models.sh             # 3 workers/cores (Rockhill shared-server limit)
#   JOBS=8 bash src/run_models.sh      # own machine, use more cores
#
# Needs data/processed/ (run preprocess.py or run_benchmarks.sh first).
# Everything printed is saved to results/logs/<host>-<date>-models/ with the result csv files.

set -euo pipefail

JOBS="${JOBS:-3}"
RUN_DIR="results/logs/$(hostname -s)-$(date +%Y%m%d-%H%M)-models"
mkdir -p "$RUN_DIR"

if [ ! -f data/processed/train.parquet ] || [ ! -f data/processed/test.parquet ]; then
    echo "No data/processed/train.parquet or test.parquet. Run python src/preprocess.py first." >&2
    exit 1
fi

# record where these results came from
{
    echo "host: $(hostname)"
    echo "date: $(date)"
    echo "git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    python --version
    echo "cores (logical): $(getconf _NPROCESSORS_ONLN)"
    echo "jobs: $JOBS"
} | tee "$RUN_DIR/environment.txt"

run() {   # run <log name> <command...>: show output live and save it
    local name="$1"; shift
    echo; echo ">>> $*"
    "$@" 2>&1 | tee "$RUN_DIR/$name.log"
}

run tune     python -u src/tune_models.py --workers "$JOBS"
# OMP_NUM_THREADS caps the hist gradient boosting threads, --jobs the random forest processes
run train    env OMP_NUM_THREADS="$JOBS" python -u src/train.py --jobs "$JOBS"
run evaluate python -u src/evaluate.py

cp results/model_tuning.csv results/best_params.json results/train_times.csv results/metrics.csv "$RUN_DIR/"
echo; echo "All done. Logs and results saved in $RUN_DIR"
