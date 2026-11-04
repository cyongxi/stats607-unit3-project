
"""Check numerical stability of the Lasso simulation."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.dgps import generate_data
from src.methods import fit_lasso
from src.scenarios import generate_scenarios


MASTER_SEED = 6072026
REPLICATIONS = 10
THETA_VALUES = (0.2, 1.0, 2.4)

OUTPUT_DIR = Path("results/benchmarks")


def main():
    scenarios = generate_scenarios()
    selected = scenarios[
        scenarios["theta"].isin(THETA_VALUES)
    ]

    seed_sequence = np.random.SeedSequence(MASTER_SEED)
    seeds = iter(
        seed_sequence.spawn(len(selected) * REPLICATIONS)
    )

    records = []
    issues = []

    for _, scenario in selected.iterrows():
        for replication in range(REPLICATIONS):
            rng = np.random.default_rng(next(seeds))

            status = "success"
            warning_count = 0
            convergence_count = 0
            n_iter = np.nan
            dual_gap = np.nan

            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")

                    data = generate_data(
                        n=int(scenario["n"]),
                        p=int(scenario["p"]),
                        k=int(scenario["k"]),
                        sigma=float(scenario["sigma"]),
                        beta_min=float(scenario["beta_min"]),
                        rng=rng,
                    )

                    fitted = fit_lasso(
                        X=data["X"],
                        y=data["y"],
                        k=int(scenario["k"]),
                        sigma=float(scenario["sigma"]),
                    )

                warning_count = len(caught)
                convergence_count = sum(
                    issubclass(w.category, ConvergenceWarning)
                    for w in caught
                )

                for w in caught:
                    issues.append({
                        "scenario_id": scenario["scenario_id"],
                        "replication": replication,
                        "type": w.category.__name__,
                        "message": str(w.message),
                    })

                n_iter = fitted["n_iter"]
                dual_gap = fitted["dual_gap"]

            except Exception as exc:
                status = "exception"
                issues.append({
                    "scenario_id": scenario["scenario_id"],
                    "replication": replication,
                    "type": type(exc).__name__,
                    "message": str(exc),
                })

            records.append({
                "scenario_id": scenario["scenario_id"],
                "replication": replication,
                "theta": scenario["theta"],
                "p": scenario["p"],
                "k": scenario["k"],
                "n": scenario["n"],
                "status": status,
                "warnings": warning_count,
                "convergence_warnings": convergence_count,
                "n_iter": n_iter,
                "dual_gap": dual_gap,
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(records)
    results.to_csv(
        OUTPUT_DIR / "stability_check_results.csv",
        index=False,
    )

    issue_columns = [
        "scenario_id", "replication", "type", "message"
    ]
    pd.DataFrame(
        issues, columns=issue_columns
    ).to_csv(
        OUTPUT_DIR / "stability_check_issues.csv",
        index=False,
    )

    print("Total runs:", len(results))
    print("Warnings:", int(results["warnings"].sum()))
    print(
        "Convergence warnings:",
        int(results["convergence_warnings"].sum()),
    )
    print(
        "Exceptions:",
        int((results["status"] == "exception").sum()),
    )
    print(
        "Reached max_iter:",
        int((results["n_iter"] >= 20000).sum()),
    )
    print(
        "Non-finite dual gaps:",
        int(
            (
                (results["status"] == "success")
                & ~np.isfinite(results["dual_gap"])
            ).sum()
        ),
    )


if __name__ == "__main__":
    main()
