
"""Visualize controlled baseline-versus-optimized benchmarks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BENCHMARK_DIR = Path("results/benchmarks")
FIGURE_DIR = Path("results/figures")

raw = pd.read_csv(
    BENCHMARK_DIR / "controlled_benchmark_raw.csv"
)

summary = pd.read_csv(
    BENCHMARK_DIR / "controlled_benchmark_summary.csv"
)

order = [
    "original",
    "algorithmic_only",
    "parallel_only_4",
    "combined_2",
    "combined_4",
]

labels = [
    "Original",
    "Algorithmic",
    "Parallel (4)",
    "Combined (2)",
    "Combined (4)",
]

summary = summary.set_index("configuration").loc[order]
assert raw.groupby("configuration").size().reindex(order).eq(3).all()

x = np.arange(len(order))

# Figure 1: Real runtime and variation.
fig, ax = plt.subplots(figsize=(10, 5.5))

medians = summary["median_real"].to_numpy()
minimum = summary["min_real"].to_numpy()
maximum = summary["max_real"].to_numpy()

bars = ax.bar(x, medians, alpha=0.85)

ax.errorbar(
    x,
    medians,
    yerr=[medians - minimum, maximum - medians],
    fmt="none",
    ecolor="black",
    capsize=5,
)

for bar, speedup in zip(bars, summary["speedup"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"{speedup:.2f}x",
        ha="center",
    )

ax.set_xticks(x, labels)
ax.set_ylabel("Median wall-clock runtime (seconds)")
ax.set_title("Original vs Optimized Simulation Performance")
ax.set_ylim(0, max(maximum) * 1.2)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(
    FIGURE_DIR / "controlled_runtime_comparison.png",
    dpi=200,
)
plt.close(fig)

# Figure 2: CPU resource trade-off.
fig, ax = plt.subplots(figsize=(10, 5.5))

width = 0.35
ax.bar(
    x - width / 2,
    summary["median_real"],
    width,
    label="Wall-clock time",
)

ax.bar(
    x + width / 2,
    summary["median_cpu"],
    width,
    label="Aggregate CPU time",
)

ax.set_xticks(x, labels)
ax.set_ylabel("Time (seconds)")
ax.set_title("Runtime vs CPU Resource Consumption")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "controlled_cpu_tradeoff.png",
    dpi=200,
)
plt.close(fig)

print("Saved:")
print(FIGURE_DIR / "controlled_runtime_comparison.png")
print(FIGURE_DIR / "controlled_cpu_tradeoff.png")
