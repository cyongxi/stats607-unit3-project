
"""Regression tests for Unit 3 optimization."""

import numpy as np
import pytest

from src.dgps import generate_data
from src.methods import calculate_lambda, fit_lasso
from src.metrics import exact_recovery


@pytest.mark.parametrize(
    "n,p,k",
    [
        (100, 128, 10),
        (200, 256, 30),
        (400, 512, 60),
    ],
)
def test_dgp_response_consistency(n, p, k):
    """Check that generated responses match the linear model."""
    rng = np.random.default_rng(6072026)

    data = generate_data(
        n=n,
        p=p,
        k=k,
        sigma=0.5,
        beta_min=0.5,
        rng=rng,
    )

    X = data["X"]
    y = data["y"]
    beta = data["beta_true"]

    # Reconstruct the exact random noise using the same random stream.
    reference_rng = np.random.default_rng(6072026)
    reference_rng.normal(size=(n, p))
    reference_rng.choice(p, size=k, replace=False)
    reference_rng.choice([-1.0, 1.0], size=k)
    noise = reference_rng.normal(scale=0.5, size=n)

    expected_y = X @ beta + noise

    np.testing.assert_allclose(
        y,
        expected_y,
        rtol=1e-10,
        atol=1e-10,
    )


def test_lasso_recovery_is_reproducible():
    """Identical random seeds should produce identical recovery results."""
    seed = 6072026

    outputs = []

    for _ in range(2):
        rng = np.random.default_rng(seed)

        data = generate_data(
            n=200,
            p=128,
            k=10,
            sigma=0.5,
            beta_min=0.5,
            rng=rng,
        )

        fit = fit_lasso(
            X=data["X"],
            y=data["y"],
            k=10,
            sigma=0.5,
        )

        outputs.append({
            "beta_hat": fit["beta_hat"],
            "success": exact_recovery(
                fit["beta_hat"],
                data["beta_true"],
            ),
        })

    np.testing.assert_allclose(
        outputs[0]["beta_hat"],
        outputs[1]["beta_hat"],
        rtol=1e-12,
        atol=1e-12,
    )

    assert outputs[0]["success"] == outputs[1]["success"]


def test_parallel_matches_serial_across_scenarios():
    """Parallel and serial results should match across scenarios."""
    from src.parallel import run_parallel
    from src.simulation import run_simulation
    from src.scenarios import generate_scenarios

    # Select scenarios with different dimensions and regimes.
    scenarios = generate_scenarios().iloc[[0, 42, 84]]

    serial = run_simulation(
        scenarios,
        n_replications=3,
        master_seed=6072026,
    )

    parallel = run_parallel(
        scenarios,
        n_replications=3,
        master_seed=6072026,
        max_workers=2,
    )

    import pandas as pd

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
    """Precomputing lambda must not change Lasso estimates."""
    rng = np.random.default_rng(6072026)

    data = generate_data(
        n=n,
        p=p,
        k=k,
        sigma=0.5,
        beta_min=0.5,
        rng=rng,
    )

    # Original behavior: calculate lambda inside fit_lasso().
    original_fit = fit_lasso(
        X=data["X"],
        y=data["y"],
        k=k,
        sigma=0.5,
    )

    # Optimized behavior: calculate lambda once and pass it in.
    lam = calculate_lambda(n, p, k, sigma=0.5)

    optimized_fit = fit_lasso(
        X=data["X"],
        y=data["y"],
        k=k,
        sigma=0.5,
        lam=lam,
    )

    np.testing.assert_allclose(
        original_fit["beta_hat"],
        optimized_fit["beta_hat"],
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        original_fit["lambda_n"],
        optimized_fit["lambda_n"],
        rtol=1e-12,
        atol=1e-12,
    )

    assert original_fit["n_iter"] == optimized_fit["n_iter"]

    np.testing.assert_allclose(
        original_fit["dual_gap"],
        optimized_fit["dual_gap"],
        rtol=1e-10,
        atol=1e-10,
    )

    original_success = exact_recovery(
        original_fit["beta_hat"],
        data["beta_true"],
    )

    optimized_success = exact_recovery(
        optimized_fit["beta_hat"],
        data["beta_true"],
    )

    assert original_success == optimized_success
