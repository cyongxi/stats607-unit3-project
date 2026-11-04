"""Lasso estimation for reproducing Wainwright (2009), Figure 2."""

import math

import numpy as np
from sklearn.linear_model import Lasso


def calculate_lambda(n, p, k, sigma):
    """Calculate the regularization parameter from Wainwright (2009)."""
    if not (1 < k < p):
        raise ValueError("Expected 1 < k < p.")

    if n <= 0:
        raise ValueError("n must be positive.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    return math.sqrt(
        2 * sigma**2 * math.log(k) * math.log(p - k) / n
    )


def fit_lasso(X, y, k, sigma, tol=1e-7, max_iter=20000, lam=None):
    """Fit Lasso using the regularization parameter from the paper."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix.")

    n, p = X.shape

    if y.shape != (n,):
        raise ValueError("y must have length n.")

    if lam is None:
        lam = calculate_lambda(n, p, k, sigma)

    model = Lasso(
        alpha=lam,
        fit_intercept=False,
        max_iter=max_iter,
        tol=tol,
        selection="cyclic",
    )

    model.fit(X, y)

    return {
        "beta_hat": model.coef_.copy(),
        "lambda_n": lam,
        "n_iter": model.n_iter_,
        "dual_gap": model.dual_gap_,
    }
