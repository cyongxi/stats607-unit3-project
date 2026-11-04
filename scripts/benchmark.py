
"""Reproducible baseline-versus-optimized simulation benchmarks."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "stats607-unit3-baseline"
OUTPUT_DIR = ROOT / "results" / "benchmarks"

RAW_PATH = OUTPUT_DIR / "controlled_benchmark_raw.csv"
SUMMARY_PATH = OUTPUT_DIR / "controlled_benchmark_summary.csv"

REPEATS = 3

CONFIGURATIONS = {
    "original": (BASELINE, ["-m", "src.simulation"]),
    "algorithmic_only": (ROOT, ["-m", "src.simulation"]),
    "parallel_only_4": (
        ROOT, ["-m", "src.parallel_only", "--workers", "4"]
    ),
    "combined_2": (
        ROOT, ["-m", "src.parallel", "--workers", "2"]
    ),
    "combined_4": (
        ROOT, ["-m", "src.parallel", "--workers", "4"]
    ),
}

ORDER = list(CONFIGURATIONS)


def check_setup():
    """Validate the independent baseline and required sources."""
    if not BASELINE.is_dir():
        raise FileNotFoundError(
            f"Baseline worktree not found: {BASELINE}"
        )

    baseline_commit = subprocess.check_output(
        ["git", "-C", str(BASELINE), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    expected_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "unit3-baseline"],
        text=True,
    ).strip()

    if baseline_commit != expected_commit:
        raise RuntimeError(
            "Baseline worktree is not at the unit3-baseline tag."
        )

    for filename in (
        "src/simulation.py",
        "src/parallel.py",
        "src/parallel_only.py",
    ):
        if not (ROOT / filename).is_file():
            raise FileNotFoundError(ROOT / filename)


def run_one(name, repeat, env):
    """Run one full simulation and capture time(1) measurements."""
    cwd, arguments = CONFIGURATIONS[name]
    command = ["/usr/bin/time", "-p", sys.executable] + arguments
    logfile = OUTPUT_DIR / f"controlled_auto_{name}_run{repeat}.log"

    print(f"Running {name}: {repeat}/{REPEATS}", flush=True)

    with logfile.open("w") as output:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )

    content = logfile.read_text()

    if "Completed 25200 replications." not in content:
        raise RuntimeError(f"Incomplete simulation: {logfile}")

    values = {}
    for metric in ("real", "user", "sys"):
        matches = re.findall(
            rf"^{metric}\s+([0-9.]+)\s*$",
            content,
            flags=re.MULTILINE,
        )
        if len(matches) != 1:
            raise ValueError(f"Invalid {metric} in {logfile}")
        values[metric] = float(matches[0])

    print(f"  Wall time: {values['real']:.2f}s", flush=True)

    return {
        "configuration": name,
        "run": repeat,
        **values,
        "cpu_total": values["user"] + values["sys"],
    }


def summarize(raw):
    """Calculate runtime, CPU cost, and baseline-relative speedup."""
    expected = {
        (name, repeat)
        for name in ORDER
        for repeat in range(1, REPEATS + 1)
    }
    observed = set(
        zip(raw["configuration"], raw["run"])
    )

    if observed != expected or len(raw) != len(expected):
        raise ValueError("Expected exactly 15 benchmark records.")

    summary = (
        raw.groupby("configuration")
        .agg(
            median_real=("real", "median"),
            mean_real=("real", "mean"),
            min_real=("real", "min"),
            max_real=("real", "max"),
            median_cpu=("cpu_total", "median"),
        )
        .reindex(ORDER)
        .reset_index()
    )

    baseline = summary.loc[
        summary["configuration"] == "original",
        "median_real",
    ].iloc[0]

    summary["speedup"] = baseline / summary["median_real"]
    summary["runtime_reduction_pct"] = (
        100 * (1 - summary["median_real"] / baseline)
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate reports from existing controlled CSV.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        if not RAW_PATH.is_file():
            raise FileNotFoundError(RAW_PATH)
        print("Using existing controlled benchmark results.")
        raw = pd.read_csv(RAW_PATH)

    else:
        check_setup()

        env = os.environ.copy()
        for key in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            env[key] = "1"

        records = []

        # Rotate configuration order across repetitions.
        for repeat in range(1, REPEATS + 1):
            offset = (repeat - 1) % len(ORDER)
            run_order = ORDER[offset:] + ORDER[:offset]

            for name in run_order:
                records.append(run_one(name, repeat, env))

        raw = pd.DataFrame(records)
        raw.to_csv(RAW_PATH, index=False)

    summary = summarize(raw)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nControlled Benchmark Summary:")
    print(summary.round(3).to_string(index=False))

    subprocess.run(
        [sys.executable, "-m", "scripts.plot_controlled_benchmark"],
        cwd=ROOT,
        check=True,
    )

    print("\nBenchmark reporting completed.")


if __name__ == "__main__":
    main()
