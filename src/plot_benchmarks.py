# plot_benchmarks.py - charts for the search and gradient benchmarks - Tudor
#
# Reads one or more run folders written by run_benchmarks.sh and saves the charts to results/plots/
#
#   python src/plot_benchmarks.py results/logs/rockhill-20261118-1150
#   python src/plot_benchmarks.py results/logs/<run A> results/logs/<run B>    # compare machines

import argparse
import os

import matplotlib
matplotlib.use("Agg")   # draw straight to files, works on the server too
import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = "./results/plots"
TASKS = {"search": "Hyperparameter search", "gradient": "Gradient computation"}

# sized for a printed poster
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 11, "figure.titlesize": 16,
    "axes.spines.top": False, "axes.spines.right": False,
})


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


def source(runs):
    # where the numbers come from, shown under every chart title
    repeats = int(timings(runs[0][1], "search")["runs"].max())
    machines = ", ".join(label for label, _ in runs)
    return f"{machines}  ·  mean of {repeats} runs per point"


def plot_speedup(runs):
    # speedup against worker count for each machine, with the ideal (linear) line
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, (task, title) in zip(axes, TASKS.items()):
        top = 1
        for i, (label, run_dir) in enumerate(runs):
            t = timings(run_dir, task)
            line, = ax.plot(t["workers"], t["speedup"], marker="o", markersize=8, linewidth=2.5, label=label)
            # label each point (not 1 worker); alternate below/above so two machines don't overlap
            offset = (10, -30) if i % 2 == 0 else (-60, 10)
            for w, s, e in zip(t["workers"], t["speedup"], t["efficiency"]):
                if w > 1:
                    ax.annotate(f"{s:.2f}x ({e:.0%})", (w, s), textcoords="offset points",
                                xytext=offset, fontsize=11, color=line.get_color(), fontweight="bold")
            top = max(top, t["workers"].max())
        ax.plot([1, top], [1, top], linestyle="--", color="grey", label="Ideal (linear)")
        ax.set(title=title, xlabel="Workers", xticks=range(1, top + 1), xlim=(0.8, top + 0.6))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Speedup vs 1 worker")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Speedup: sequential vs parallel\n{source(runs)}  ·  labels: speedup (efficiency)",
                 fontsize=14)
    return fig


def plot_times(runs):
    # mean wall-clock time per worker count, error bars = standard deviation over repeats
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    width = 0.8 / len(runs)
    for ax, (task, title) in zip(axes, TASKS.items()):
        for i, (label, run_dir) in enumerate(runs):
            t = timings(run_dir, task)
            x = t["workers"] + (i - (len(runs) - 1) / 2) * width
            bars = ax.bar(x, t["wall_mean"], width, yerr=t["wall_std"].fillna(0), capsize=4, label=label)
            ax.bar_label(bars, labels=[f"{v:.1f} s" for v in t["wall_mean"]], label_type="center",
                         color="white", fontweight="bold", fontsize=11)
        ax.set(title=title, xlabel="Workers", ylabel="Wall-clock time (s)",
               xticks=sorted(timings(runs[0][1], task)["workers"]))
        ax.grid(axis="y", alpha=0.3)
    if len(runs) > 1:
        axes[0].legend()
    fig.suptitle(f"Mean run time per worker count\n{source(runs)}  ·  error bars: std over repeats",
                 fontsize=14)
    return fig


def plot_longest_first(runs):
    # load imbalance: grid order vs longest-first at the same worker count, against the ideal time
    fig, ax = plt.subplots(figsize=(8, 5.2))
    rows = []
    for label, run_dir in runs:
        lf = pd.read_csv(os.path.join(run_dir, "search_timings_longest_first.csv")).iloc[0]
        grid = timings(run_dir, "search")
        g = grid[grid["workers"] == lf["workers"]].iloc[0]
        rows.append((label, g["wall_mean"], lf["wall_mean"], lf["busy_mean"] / lf["workers"], int(lf["workers"])))

    w = 0.27
    x = list(range(len(rows)))
    grid_bars = ax.bar([i - w for i in x], [r[1] for r in rows], w, label="Grid order")
    lf_bars = ax.bar(x, [r[2] for r in rows], w, label="Longest first")
    ideal_bars = ax.bar([i + w for i in x], [r[3] for r in rows], w, color="lightgrey",
                        label="Ideal (total work ÷ workers)")
    ax.bar_label(grid_bars, labels=[f"{r[1]:.1f} s" for r in rows], padding=3, fontsize=11)
    ax.bar_label(lf_bars, labels=[f"{r[2]:.1f} s\n(−{1 - r[2] / r[1]:.0%})" for r in rows],
                 padding=3, fontsize=11, fontweight="bold")
    ax.bar_label(ideal_bars, labels=[f"{r[3]:.1f} s" for r in rows], padding=3, fontsize=11)

    names = [f"{r[4]} workers" if len(rows) == 1 else f"{r[0]}\n({r[4]} workers)" for r in rows]
    ax.set(xticks=x, xticklabels=names, ylabel="Wall-clock time (s)",
           ylim=(0, max(r[1] for r in rows) * 1.45))
    ax.set_title(f"Hyperparameter search: load imbalance\n{source(runs)}", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_convergence(runs):
    # training RMSE per gradient descent step (identical for every worker count)
    label, run_dir = runs[0]
    h = pd.read_csv(os.path.join(run_dir, "gradient_history.csv"))
    t = timings(run_dir, "gradient")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(h["step"], h["train_rmse"], linewidth=2.5)
    first, last = h.iloc[0], h.iloc[-1]
    ax.annotate(f"{first['train_rmse']:.1f} min", (first["step"], first["train_rmse"]),
                textcoords="offset points", xytext=(10, -5), fontsize=11)
    ax.annotate(f"{last['train_rmse']:.2f} min after {int(last['step'])} steps",
                (last["step"], last["train_rmse"]), textcoords="offset points", xytext=(-10, 12),
                ha="right", fontsize=11, fontweight="bold")
    if t["all_identical"].all():
        workers = ", ".join(str(w) for w in t["workers"])
        ax.text(0.98, 0.6, f"Weights bit-identical\nfor {workers} workers",
                transform=ax.transAxes, ha="right", fontsize=11, color="dimgrey")
    ax.set(xlabel="Step", ylabel="Training RMSE (minutes)")
    ax.set_title(f"Gradient descent convergence (data-parallel, shard sums in fixed order)\n{label}",
                 fontsize=14)
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
