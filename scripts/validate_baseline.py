
"""Validate optimized simulation against the original Unit 2 baseline."""

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
SCENARIO_INDICES = (0, 42, 84)


# Execute this code in separate Python processes so that the
# original and optimized versions never share imported modules.
RUNNER = """
import sys
from pathlib import Path

from src.scenarios import generate_scenarios
from src.simulation import run_simulation

indices = [int(value) for value in sys.argv[2:]]
scenarios = generate_scenarios().iloc[indices]

results = run_simulation(
    scenarios=scenarios,
    n_replications=3,
    master_seed=6072026,
)

results.to_csv(Path(sys.argv[1]), index=False)
"""


def git_commit(directory, reference):
    """Return the commit associated with a Git reference."""
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", reference],
        text=True,
    ).strip()


def validate_baseline():
    """Compare original and optimized replication-level results."""
    if not BASELINE.is_dir():
        raise FileNotFoundError(
            f"Baseline worktree not found: {BASELINE}"
        )

    expected_commit = git_commit(ROOT, "unit3-baseline")
    actual_commit = git_commit(BASELINE, "HEAD")

    if actual_commit != expected_commit:
        raise RuntimeError(
            "Baseline worktree does not match unit3-baseline tag."
        )

    env = os.environ.copy()

    # Prevent a caller-supplied PYTHONPATH from importing
    # source files from the wrong worktree.
    env.pop("PYTHONPATH", None)

    with tempfile.TemporaryDirectory() as temp_dir:
        temporary = Path(temp_dir)

        frames = {}

        for label, directory in (
            ("Original Unit 2", BASELINE),
            ("Optimized Unit 3", ROOT),
        ):
            output_path = temporary / (
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
                    *map(str, SCENARIO_INDICES),
                ],
                cwd=directory,
                env=env,
                check=True,
            )

            frames[label] = pd.read_csv(output_path)

        original = frames["Original Unit 2"]
        optimized = frames["Optimized Unit 3"]

        expected_rows = len(SCENARIO_INDICES) * N_REPLICATIONS

        assert len(original) == expected_rows
        assert len(optimized) == expected_rows

        pd.testing.assert_frame_equal(
            original.reset_index(drop=True),
            optimized.reset_index(drop=True),
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )

        print("\nBASELINE REGRESSION VALIDATION PASSED")
        print(f"Baseline commit: {actual_commit[:7]}")
        print(f"Scenarios checked: {len(SCENARIO_INDICES)}")
        print(f"Replications per scenario: {N_REPLICATIONS}")
        print(f"Records compared: {expected_rows}")
        print("All output columns agree within tolerance.")


if __name__ == "__main__":
    validate_baseline()
