"""Regression tests for Unit 3 optimization."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from src.dgps import generate_data
from src.methods import calculate_lambda, fit_lasso
from src.metrics import exact_recovery
from src.parallel import run_parallel
from src.scenarios import generate_scenarios
from src.simulation import run_simulation


MASTER_SEED = 6072026


def representative_scenarios():
    """Select nine scenarios by factor values rather than row positions."""
    design = generate_scenarios()

    # Each regime covers all three dimensions and all three theta values.
    settings = [
        ("linear", 128, 0.2),
        ("linear", 256, 1.0),
        ("linear", 512, 2.4),
        ("sublinear", 128, 1.0),
        ("sublinear", 256, 2.4),
        ("sublinear", 512, 0.2),
        ("fractional", 128, 2.4),
        ("fractional", 256, 0.2),
        ("fractional", 512, 1.0),
    ]

    selected = []
    for regime, p, theta in settings:
        matches = design.loc[
            design["regime"].eq(regime)
            & design["p"].eq(p)
            & np.isclose(design["theta"], theta)
        ]
        assert len(matches) == 1, (regime, p, theta)
        selected.append(matches)

    scenarios = pd.concat(selected, ignore_index=True)

    assert len(scenarios) == 9
    assert scenarios["scenario_id"].is_unique
    assert set(
        zip(scenarios["regime"], scenarios["p"])
    ) == set(
        product(
            ("linear", "sublinear", "fractional"),
            (128, 256, 512),
        )
    )
    assert set(scenarios["theta"]) == {0.2, 1.0, 2.4}

    return scenarios


@pytest.mark.parametrize(
    "n,p,k",
    [
        (100, 128, 10),
        (200, 256, 30),
        (400, 512, 60),
    ],
)
def test_dgp_response_consistency(n, p, k):
    """Generated responses must satisfy the specified linear model."""
    data = generate_data(
        n=n,
        p=p,
        k=k,
        sigma=0.5,
        beta_min=0.5,
        rng=np.random.default_rng(MASTER_SEED),
    )

    # Reconstruct noise from the documented sequence of RNG operations.
    reference_rng = np.random.default_rng(MASTER_SEED)
    reference_rng.normal(size=(n, p))
    reference_rng.choice(p, size=k, replace=False)
    reference_rng.choice([-1.0, 1.0], size=k)
    noise = reference_rng.normal(scale=0.5, size=n)

    np.testing.assert_allclose(
        data["y"],
        data["X"] @ data["beta_true"] + noise,
        rtol=1e-10,
        atol=1e-10,
    )


def test_lasso_recovery_is_reproducible():
    """Identical seeds must reproduce coefficients and recovery outcomes."""
    outputs = []

    for _ in range(2):
        data = generate_data(
            n=200,
            p=128,
            k=10,
            sigma=0.5,
            beta_min=0.5,
            rng=np.random.default_rng(MASTER_SEED),
        )
        fitted = fit_lasso(
            X=data["X"],
            y=data["y"],
            k=10,
            sigma=0.5,
        )
        outputs.append(
            (
                fitted["beta_hat"],
                exact_recovery(
                    fitted["beta_hat"], data["beta_true"]
                ),
            )
        )

    np.testing.assert_allclose(
        outputs[0][0], outputs[1][0],
        rtol=1e-12, atol=1e-12,
    )
    assert outputs[0][1] == outputs[1][1]


def test_parallel_matches_serial_across_scenarios():
    """Serial and parallel outputs must agree across representative settings."""
    scenarios = representative_scenarios()

    serial = run_simulation(
        scenarios,
        n_replications=3,
        master_seed=MASTER_SEED,
    )
    parallel = run_parallel(
        scenarios,
        n_replications=3,
        master_seed=MASTER_SEED,
        max_workers=2,
    )

    assert len(serial) == len(parallel) == 27
    assert not serial.duplicated(
        ["scenario_id", "replication_id"]
    ).any()
    assert serial.groupby("scenario_id").size().eq(3).all()

    pd.testing.assert_frame_equal(
        serial.reset_index(drop=True),
        parallel.reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "n,p,k",
    [
        (100, 128, 10),
        (200, 256, 30),
        (400, 512, 60),
    ],
)
def test_precomputed_lambda_matches_original_lasso(n, p, k):
    """Precomputing lambda must preserve estimates and solver diagnostics."""
    data = generate_data(
        n=n,
        p=p,
        k=k,
        sigma=0.5,
        beta_min=0.5,
        rng=np.random.default_rng(MASTER_SEED),
    )

    original = fit_lasso(
        X=data["X"], y=data["y"], k=k, sigma=0.5,
    )
    optimized = fit_lasso(
        X=data["X"],
        y=data["y"],
        k=k,
        sigma=0.5,
        lam=calculate_lambda(n, p, k, sigma=0.5),
    )

    for key in ("beta_hat", "lambda_n", "dual_gap"):
        np.testing.assert_allclose(
            original[key],
            optimized[key],
            rtol=1e-10,
            atol=1e-10,
        )

    assert original["n_iter"] == optimized["n_iter"]
    assert exact_recovery(
        original["beta_hat"], data["beta_true"]
    ) == exact_recovery(
        optimized["beta_hat"], data["beta_true"]
    )