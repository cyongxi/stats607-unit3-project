
"""Empirical computational complexity analysis for the Unit 2 baseline."""

import os

# Keep BLAS threading consistent for reproducible timing.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dgps import generate_data
from src.methods import fit_lasso
from src.scenarios import (
    BETA_MIN,
    NOISE_SIGMA,
    calculate_sparsity,
)


P = 256
K = calculate_sparsity(P, "linear")
N_VALUES = (100, 200, 400, 800, 1600)
N_REPEATS = 10
MASTER_SEED = 6072026

OUTPUT_DIR = Path("results/benchmarks")
FIGURE_DIR = Path("results/figures")


def measure_complexity():
    """Measure DGP and Lasso runtime as sample size increases."""
    rows = []
    seed_sequence = np.random.SeedSequence(MASTER_SEED)
    child_seeds = iter(
        seed_sequence.spawn(len(N_VALUES) * N_REPEATS)
    )

    for n in N_VALUES:
        for repeat in range(N_REPEATS):
            rng = np.random.default_rng(next(child_seeds))

            start = time.perf_counter()
            data = generate_data(
                n=n,
                p=P,
                k=K,
                sigma=NOISE_SIGMA,
                beta_min=BETA_MIN,
                rng=rng,
            )
            dgp_time = time.perf_counter() - start

            start = time.perf_counter()
            fitted = fit_lasso(
                X=data["X"],
                y=data["y"],
                k=K,
                sigma=NOISE_SIGMA,
            )
            fit_time = time.perf_counter() - start

            rows.append({
                "n": n,
                "p": P,
                "k": K,
                "repeat": repeat,
                "dgp_time": dgp_time,
                "fit_time": fit_time,
                "total_time": dgp_time + fit_time,
                "n_iter": fitted["n_iter"],
                "dual_gap": fitted["dual_gap"],
            })

        print(f"Completed n={n}", flush=True)

    return pd.DataFrame(rows)


def summarize_and_plot(results):
    """Summarize runtimes and estimate empirical scaling exponents."""
    columns = ["dgp_time", "fit_time", "total_time"]

    summary = (
        results.groupby("n")[columns]
        .agg(["median", "mean", "std"])
    )

    summary.columns = [
        f"{name}_{stat}" for name, stat in summary.columns
    ]
    summary = summary.reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    for column, label in [
        ("dgp_time", "Data generation"),
        ("fit_time", "Lasso fitting"),
        ("total_time", "Combined"),
    ]:
        median_column = f"{column}_median"
        x = np.log(summary["n"].to_numpy())
        y = np.log(summary[median_column].to_numpy())

        slope, intercept = np.polyfit(x, y, 1)

        ax.plot(
            summary["n"],
            summary[median_column],
            marker="o",
            label=f"{label} (slope={slope:.2f})",
        )

        print(f"{label}: empirical exponent = {slope:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Median runtime (seconds)")
    ax.set_title("Baseline Computational Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "baseline_complexity.png",
        dpi=200,
    )
    plt.close(fig)

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results = measure_complexity()
    results.to_csv(
        OUTPUT_DIR / "baseline_complexity_raw.csv",
        index=False,
    )

    summary = summarize_and_plot(results)
    summary.to_csv(
        OUTPUT_DIR / "baseline_complexity_summary.csv",
        index=False,
    )

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
