"""Constrained risk budgeting under long-only position limits; binding caps can prevent equal risk contributions.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np
import pandas as pd

from .ratio_solver import checked_covariance


def _inputs(returns, cov):
    names = None
    if cov is None:
        if returns is None:
            raise ValueError("Specify returns or cov")
        if isinstance(returns, pd.DataFrame):
            names = returns.columns
        X = np.asarray(returns, float)
        if np.isnan(X).any():
            raise ValueError("returns contains missing values")
        S = np.cov(X, rowvar=False)
    else:
        if isinstance(cov, pd.DataFrame):
            names = cov.columns
        S = np.asarray(cov, float)
    return S, names


def constrained_risk_budgeting(cov, cap=1.0, max_sweeps=2000, tol=1e-10, return_info=False):
    S = checked_covariance(cov)
    n = len(S)
    if not np.isfinite(cap) or cap <= 0 or cap * n < 1 - 1e-12:
        raise ValueError("cap must define a nonempty capped simplex")
    cap = min(float(cap), 1.0)
    if abs(cap * n - 1) < 1e-12:
        w = np.full(n, 1 / n)
        info = {"method": "singleton_feasible_set", "budget_residual": 0.0}
        return (w, info) if return_info else w
    scale = np.trace(S) / n
    Q = S / scale
    b = 1.0 / n
    diagonal = Q.diagonal()
    total_sweeps = 0

    def inner(alpha, start):
        nonlocal total_sweeps
        x = start.copy()
        for iteration in range(max_sweeps):
            for i in range(n):
                beta = float(Q[i] @ x) - diagonal[i] * x[i]
                root = np.sqrt(beta * beta + 4 * diagonal[i] * alpha * b)
                # This algebraic form avoids subtractive cancellation.
                value = (
                    2 * alpha * b / (beta + root)
                    if beta >= 0
                    else (root - beta) / (2 * diagonal[i])
                )
                x[i] = min(cap, value)
            gradient = Q @ x - alpha * b / x
            free = x < cap - 1e-12
            error = max(
                float(np.max(np.abs(gradient[free]), initial=0)),
                float(np.max(np.maximum(gradient[~free], 0), initial=0)),
            )
            relative = error / max(1.0, np.max(np.abs(Q @ x)), np.max(alpha * b / x))
            if relative <= tol * 0.01:
                total_sweeps += iteration + 1
                return x, relative
        raise RuntimeError("Constrained risk-budgeting inner KKT test did not converge")

    # At this upper alpha, x=cap*1 satisfies all upper-bound KKT inequalities.
    upper = 2 * max(float(np.max(cap * cap * Q.sum(axis=1) / b)), 1.0)
    lower = 0.0
    x = np.full(n, 1 / n)
    for iteration in range(100):
        alpha = (lower + upper) / 2
        x, kkt = inner(alpha, x)
        error = float(x.sum() - 1)
        if abs(error) <= tol:
            break
        if error > 0:
            upper = alpha
        else:
            lower = alpha
    else:
        raise RuntimeError("Constrained risk-budgeting budget calibration did not converge")
    # No ex-post normalization or projection: cap and KKT refer to this x.
    info = {
        "method": "box_risk_budgeting_ccd_bisection",
        "alpha": alpha * scale,
        "lambda": alpha * scale / np.sqrt(x @ S @ x),
        "relative_inner_kkt_residual": kkt,
        "budget_residual": abs(error),
        "bisection_iterations": iteration + 1,
        "coordinate_sweeps": total_sweeps,
    }
    return (x, info) if return_info else x


def erc(returns=None, cov=None, cap=1.0, sweeps=60):
    S, names = _inputs(returns, cov)
    w = constrained_risk_budgeting(S, cap=cap, max_sweeps=max(2000, sweeps))
    return pd.Series(w, names) if names is not None else w


def risk_contrib(w, cov):
    "Return risk contribution fractions RC_i / portfolio variance, summing to one."
    w = np.asarray(w, float)
    S = np.asarray(cov, float)
    rc = w * (S @ w)
    return rc / rc.sum()
