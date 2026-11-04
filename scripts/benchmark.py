
"""Benchmark serial and parallel Lasso simulations."""

import os

# Set thread limits before importing NumPy or scikit-learn.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("results/benchmarks")
FIGURE_DIR = Path("results/figures")
REPEATS = 3

CONFIGURATIONS = [
    ("serial", [sys.executable, "-m", "src.simulation"]),
    ("parallel_2", [
        sys.executable, "-m", "src.parallel", "--workers", "2"
    ]),
    ("parallel_4", [
        sys.executable, "-m", "src.parallel", "--workers", "4"
    ]),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    env = os.environ.copy()

    for repeat in range(REPEATS):
        for name, command in CONFIGURATIONS:
            print(
                f"Benchmark {name}, repeat {repeat + 1}/{REPEATS}",
                flush=True,
            )

            start = time.perf_counter()

            with open(
                OUTPUT_DIR / f"{name}_run_{repeat + 1}.log", "w"
            ) as logfile:
                subprocess.run(
                    command,
                    check=True,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

            elapsed = time.perf_counter() - start

            records.append({
                "configuration": name,
                "repeat": repeat + 1,
                "runtime_seconds": elapsed,
            })

            print(f"Completed in {elapsed:.3f} seconds")

    raw = pd.DataFrame(records)
    raw.to_csv(
        OUTPUT_DIR / "optimization_benchmark_raw.csv",
        index=False,
    )

    summary = (
        raw.groupby("configuration")["runtime_seconds"]
        .agg(
            median="median",
            mean="mean",
            min="min",
            max="max",
        )
        .reset_index()
    )

    baseline = summary.loc[
        summary["configuration"] == "serial", "median"
    ].iloc[0]

    summary["speedup"] = baseline / summary["median"]

    summary.to_csv(
        OUTPUT_DIR / "optimization_benchmark_summary.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        summary["configuration"],
        summary["median"],
    )

    ax.set_ylabel("Median wall-clock runtime (seconds)")
    ax.set_xlabel("Execution configuration")
    ax.set_title("Serial vs Parallel Simulation Runtime")

    for i, row in summary.iterrows():
        ax.text(
            i,
            row["median"] + 1,
            f'{row["median"]:.2f}s',
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "optimization_runtime_comparison.png",
        dpi=200,
    )
    plt.close(fig)

    print("\nBenchmark summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
