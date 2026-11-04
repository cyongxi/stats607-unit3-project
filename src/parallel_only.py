
"""Parallel Lasso support-recovery simulation for Unit 3."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.dgps import generate_data
from src.methods import fit_lasso
from src.metrics import exact_recovery
from src.scenarios import generate_scenarios
from src.simulation import MASTER_SEED, run_simulation


def run_scenario(task):
    """Run all replications of one scenario in a worker process."""
    scenario, seeds = task
    results = []

    n = int(scenario["n"])
    p = int(scenario["p"])
    k = int(scenario["k"])
    sigma = float(scenario["sigma"])
    beta_min = float(scenario["beta_min"])

    for replication_id, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)

        data = generate_data(
            n=n,
            p=p,
            k=k,
            sigma=sigma,
            beta_min=beta_min,
            rng=rng,
        )

        fitted = fit_lasso(
            X=data["X"],
            y=data["y"],
            k=k,
            sigma=sigma,
        )

        success = exact_recovery(
            beta_hat=fitted["beta_hat"],
            beta_true=data["beta_true"],
        )

        results.append({
            "scenario_id": scenario["scenario_id"],
            "replication_id": replication_id,
            "regime": scenario["regime"],
            "p": p,
            "k": k,
            "n": n,
            "theta": float(scenario["theta"]),
            "success": int(success),
            "lambda_n": fitted["lambda_n"],
            "n_iter": fitted["n_iter"],
            "dual_gap": fitted["dual_gap"],
        })

    return results


def run_parallel(
    scenarios,
    n_replications=None,
    master_seed=MASTER_SEED,
    max_workers=2,
):
    """Parallelize across scenarios with deterministic random streams."""
    total_replications = sum(
        n_replications
        if n_replications is not None
        else int(row["n_replications"])
        for _, row in scenarios.iterrows()
    )

    child_seeds = iter(
        np.random.SeedSequence(master_seed).spawn(total_replications)
    )

    tasks = []

    for _, scenario in scenarios.iterrows():
        reps = (
            n_replications
            if n_replications is not None
            else int(scenario["n_replications"])
        )

        seeds = [next(child_seeds) for _ in range(reps)]
        tasks.append((scenario.to_dict(), seeds))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        batches = list(executor.map(run_scenario, tasks))

    rows = [row for batch in batches for row in batch]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    scenarios = generate_scenarios()

    if args.pilot:
        scenarios = scenarios.head(1)
        n_replications = 5
        output_path = Path("results/raw/parallel_pilot.csv")
    else:
        n_replications = None
        output_path = Path("results/raw/parallel_results.csv")

    results = run_parallel(
        scenarios=scenarios,
        n_replications=n_replications,
        max_workers=args.workers,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Completed {len(results)} replications.")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
