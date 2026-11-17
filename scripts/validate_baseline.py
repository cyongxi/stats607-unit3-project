"""Compare optimized simulation with the preserved Unit 2 baseline."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "stats607-unit3-baseline"

MASTER_SEED = 6072026
N_REPLICATIONS = 3

SCENARIO_SETTINGS = (
    ("linear", 128, 0.2),
    ("linear", 256, 1.0),
    ("linear", 512, 2.4),
    ("sublinear", 128, 1.0),
    ("sublinear", 256, 2.4),
    ("sublinear", 512, 0.2),
    ("fractional", 128, 2.4),
    ("fractional", 256, 0.2),
    ("fractional", 512, 1.0),
)

# Separate processes prevent imports from mixing the two source trees.
RUNNER = """
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.scenarios import generate_scenarios
from src.simulation import run_simulation

output_path = Path(sys.argv[1])
settings = json.loads(sys.argv[2])
replications = int(sys.argv[3])
master_seed = int(sys.argv[4])

design = generate_scenarios()
selected = []

for regime, p, theta in settings:
    matches = design.loc[
        design["regime"].eq(regime)
        & design["p"].eq(p)
        & np.isclose(design["theta"], theta)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one scenario for {(regime, p, theta)}, "
            f"found {len(matches)}."
        )
    selected.append(matches)

scenarios = pd.concat(selected, ignore_index=True)
results = run_simulation(
    scenarios=scenarios,
    n_replications=replications,
    master_seed=master_seed,
)
results.to_csv(output_path, index=False)
"""


def git_output(directory, *arguments):
    """Run a read-only Git command in the specified source tree."""
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments],
        text=True,
    ).strip()


def validate_baseline():
    """Compare every recorded output column for representative scenarios."""
    if not BASELINE.is_dir():
        raise FileNotFoundError(
            f"Baseline worktree not found: {BASELINE}\n"
            "Create it with:\n"
            "git worktree add --detach "
            "../stats607-unit3-baseline unit3-baseline"
        )

    expected_commit = git_output(
        ROOT, "rev-parse", "unit3-baseline^{commit}"
    )
    actual_commit = git_output(
        BASELINE, "rev-parse", "HEAD"
    )

    if actual_commit != expected_commit:
        raise RuntimeError(
            "Baseline worktree does not match unit3-baseline."
        )

    # Generated results may change; baseline source files must not.
    source_changes = git_output(
        BASELINE,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src",
    )
    if source_changes:
        raise RuntimeError(
            "Baseline source tree contains changes:\n"
            f"{source_changes}"
        )

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        env[variable] = "1"

    with tempfile.TemporaryDirectory() as temporary:
        frames = {}

        for label, directory in (
            ("Original Unit 2", BASELINE),
            ("Optimized Unit 3", ROOT),
        ):
            output_path = Path(temporary) / (
                "original.csv"
                if directory == BASELINE
                else "optimized.csv"
            )

            print(f"Running {label}...", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    RUNNER,
                    str(output_path),
                    json.dumps(SCENARIO_SETTINGS),
                    str(N_REPLICATIONS),
                    str(MASTER_SEED),
                ],
                cwd=directory,
                env=env,
                check=True,
            )
            frames[label] = pd.read_csv(output_path)

        expected_rows = (
            len(SCENARIO_SETTINGS) * N_REPLICATIONS
        )

        for label, frame in frames.items():
            if len(frame) != expected_rows:
                raise AssertionError(
                    f"{label}: expected {expected_rows} records, "
                    f"found {len(frame)}."
                )
            if frame.duplicated(
                ["scenario_id", "replication_id"]
            ).any():
                raise AssertionError(
                    f"{label}: duplicate replication identifiers."
                )
            counts = frame.groupby("scenario_id").size()
            if (
                len(counts) != len(SCENARIO_SETTINGS)
                or not counts.eq(N_REPLICATIONS).all()
            ):
                raise AssertionError(
                    f"{label}: unexpected scenario replication counts."
                )

        pd.testing.assert_frame_equal(
            frames["Original Unit 2"].reset_index(drop=True),
            frames["Optimized Unit 3"].reset_index(drop=True),
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )

    print("\nBASELINE REGRESSION VALIDATION PASSED")
    print(f"Baseline commit: {actual_commit[:7]}")
    print(f"Scenarios checked: {len(SCENARIO_SETTINGS)}")
    print(f"Replications per scenario: {N_REPLICATIONS}")
    print(f"Records compared: {expected_rows}")
    print("Coverage: three regimes, three dimensions, three theta values.")
    print("All recorded output columns agree within tolerance.")


if __name__ == "__main__":
    validate_baseline()