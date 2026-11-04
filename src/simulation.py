"""Run reproducible Lasso support-recovery simulations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dgps import generate_data
from src.methods import calculate_lambda, fit_lasso
from src.metrics import exact_recovery
from src.scenarios import generate_scenarios


MASTER_SEED = 6072026
OUTPUT_PATH = Path("results/raw/simulation_results.csv")


def run_simulation(
    scenarios,
    n_replications=None,
    master_seed=MASTER_SEED,
):
    """Run simulation replications and return their raw results."""
    results = []

    # Create independent random streams from one master seed.
    total_replications = sum(
        n_replications
        if n_replications is not None
        else int(row["n_replications"])
        for _, row in scenarios.iterrows()
    )

    seed_sequence = np.random.SeedSequence(master_seed)
    child_seeds = iter(seed_sequence.spawn(total_replications))

    total_scenarios = len(scenarios)

    for scenario_index, (_, scenario) in enumerate(
        scenarios.iterrows(), start=1
    ):
        reps = (
            n_replications
            if n_replications is not None
            else int(scenario["n_replications"])
        )

        # Precompute scenario-specific parameters once.
        n = int(scenario["n"])
        p = int(scenario["p"])
        k = int(scenario["k"])
        sigma = float(scenario["sigma"])
        beta_min = float(scenario["beta_min"])
        theta = float(scenario["theta"])

        lam = calculate_lambda(n, p, k, sigma)

        for replication_id in range(reps):
            rng = np.random.default_rng(next(child_seeds))

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
                lam=lam,
            )

            success = exact_recovery(
                beta_hat=fitted["beta_hat"],
                beta_true=data["beta_true"],
            )

            results.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "replication_id": replication_id,
                    "regime": scenario["regime"],
                    "p": p,
                    "k": k,
                    "n": n,
                    "theta": theta,
                    "success": int(success),
                    "lambda_n": fitted["lambda_n"],
                    "n_iter": fitted["n_iter"],
                    "dual_gap": fitted["dual_gap"],
                }
            )
        print(
            f"Completed scenario {scenario_index}/{total_scenarios}: "
            f"{scenario['scenario_id']}",
            flush=True,
        )

    return pd.DataFrame(results)


def main():
    """Run either a small pilot or the full simulation study."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run a small pilot using one scenario and five replications.",
    )
    args = parser.parse_args()

    scenarios = generate_scenarios()

    if args.pilot:
        scenarios = scenarios.head(1)
        n_replications = 5
        output_path = Path("results/raw/pilot_results.csv")
    else:
        n_replications = None
        output_path = OUTPUT_PATH

    print(
        f"Running {len(scenarios)} scenarios "
        f"with master seed {MASTER_SEED}...",
        flush=True,
    )

    results = run_simulation(
        scenarios,
        n_replications=n_replications,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Completed {len(results)} replications.")
    print(f"Saved raw results to {output_path}")


if __name__ == "__main__":
    main()
