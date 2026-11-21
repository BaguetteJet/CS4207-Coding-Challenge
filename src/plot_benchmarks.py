# plot_benchmarks.py - charts for the search and gradient benchmarks - Tudor
#
# Reads one or more run folders written by run_benchmarks.sh and saves the charts to results/plots/
#
#   python src/plot_benchmarks.py results/logs/rockhill-20261118-1150
#   python src/plot_benchmarks.py results/logs/<mac folder> results/logs/rockhill-20261118-1150

import argparse
import os

import matplotlib
matplotlib.use("Agg")   # draw straight to files, works on the server too
import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = "./results/plots"
TASKS = {"search": "Hyperparameter search", "gradient": "Gradient computation"}


def machine_name(run_dir):
    # e.g. "Rockhill (AMD EPYC 9124)" or "Apple M4", from environment.txt
    lines = open(os.path.join(run_dir, "environment.txt")).read().splitlines()
    host = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("host:")), "")
    cpu = ""
    for line in lines:
        if line.startswith("Model name:"):
            cpu = line.split(":", 1)[1].replace("16-Core Processor", "").strip()
        elif line.startswith("Apple"):
            cpu = line.strip()
    if host and ":" not in host and cpu:      # a readable host name (Macs often report an address)
        return f"{host.capitalize()} ({cpu})"
    return cpu or host or os.path.basename(run_dir)


def timings(run_dir, task):
    name = "search_timings_grid.csv" if task == "search" else "gradient_timings_ordered.csv"
    return pd.read_csv(os.path.join(run_dir, name))


def plot_speedup(runs):
    # speedup against worker count for each machine, with the ideal (linear) line
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (task, title) in zip(axes, TASKS.items()):
        top = 1
        for i, (label, run_dir) in enumerate(runs):
            t = timings(run_dir, task)
            line, = ax.plot(t["workers"], t["speedup"], marker="o", label=label)
            # label each point (not 1 worker); alternate below/above so two machines don't overlap
            offset = (8, -26) if i % 2 == 0 else (-44, 8)
            for w, s, e in zip(t["workers"], t["speedup"], t["efficiency"]):
                if w > 1:
                    ax.annotate(f"{s:.2f}x ({e:.0%})", (w, s), textcoords="offset points",
                                xytext=offset, fontsize=8, color=line.get_color())
            top = max(top, t["workers"].max())
        ax.plot([1, top], [1, top], linestyle="--", color="grey", label="Ideal (linear)")
        ax.set(title=title, xlabel="Workers", xticks=range(1, top + 1), xlim=(0.8, top + 0.5))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Speedup vs 1 worker (mean of repeats)")
    axes[0].legend()
    fig.suptitle("Speedup: sequential vs parallel")
    return fig


def plot_times(runs):
    # mean wall-clock time per worker count, error bars = standard deviation over repeats
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.8 / len(runs)
    for ax, (task, title) in zip(axes, TASKS.items()):
        for i, (label, run_dir) in enumerate(runs):
            t = timings(run_dir, task)
            x = t["workers"] + (i - (len(runs) - 1) / 2) * width
            ax.bar(x, t["wall_mean"], width, yerr=t["wall_std"].fillna(0), capsize=3, label=label)
        ax.set(title=title, xlabel="Workers", ylabel="Wall-clock time (s)",
               xticks=sorted(timings(runs[0][1], task)["workers"]))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle("Mean run time (error bars: std over repeats)")
    return fig


def plot_longest_first(runs):
    # load imbalance: grid order vs longest-first at the same worker count, against the ideal time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for label, run_dir in runs:
        lf = pd.read_csv(os.path.join(run_dir, "search_timings_longest_first.csv")).iloc[0]
        grid = timings(run_dir, "search")
        g = grid[grid["workers"] == lf["workers"]].iloc[0]
        rows.append((label, g["wall_mean"], lf["wall_mean"], lf["busy_mean"] / lf["workers"], int(lf["workers"])))

    x = range(len(rows))
    ax.bar([i - 0.27 for i in x], [r[1] for r in rows], 0.27, label="Grid order")
    ax.bar(list(x), [r[2] for r in rows], 0.27, label="Longest first")
    ax.bar([i + 0.27 for i in x], [r[3] for r in rows], 0.27, label="Ideal (total work / workers)", color="lightgrey")
    for i, r in enumerate(rows):
        ax.annotate(f"-{1 - r[2] / r[1]:.0%}", (i, r[2]), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)
    ax.set(xticks=list(x), xticklabels=[f"{r[0]}\n({r[4]} workers)" for r in rows],
           ylabel="Wall-clock time (s)", title="Hyperparameter search: load imbalance")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_convergence(runs):
    # training RMSE per gradient descent step (the same on every machine and worker count)
    label, run_dir = runs[0]
    h = pd.read_csv(os.path.join(run_dir, "gradient_history.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(h["step"], h["train_rmse"])
    ax.set(xlabel="Step", ylabel="Training RMSE (minutes)", title="Gradient descent convergence")
    ax.grid(alpha=0.3)
    return fig


def main():
    parser = argparse.ArgumentParser(description="plot benchmark results")
    parser.add_argument("run_dirs", nargs="+", help="folders from results/logs/")
    args = parser.parse_args()

    runs = [(machine_name(d), d) for d in args.run_dirs]
    os.makedirs(PLOT_DIR, exist_ok=True)

    for name, make in [("speedup", plot_speedup), ("times", plot_times),
                       ("load_imbalance", plot_longest_first), ("convergence", plot_convergence)]:
        fig = make(runs)
        fig.tight_layout()
        path = os.path.join(PLOT_DIR, f"{name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()
