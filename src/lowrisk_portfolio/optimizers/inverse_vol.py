"""Inverse-volatility weights projected onto the capped simplex.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np
import pandas as pd


def _proj(v, cap):
    "Project onto the capped simplex by bisection on the threshold."
    lo, hi = v.min() - 1, v.max()
    for _ in range(34):
        th = (lo + hi) / 2
        w = np.clip(v - th, 0, cap)
        if w.sum() > 1:
            lo = th
        else:
            hi = th
    return np.clip(v - (lo + hi) / 2, 0, cap)


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


def inverse_vol(returns=None, cov=None, cap=1.0):
    S, names = _inputs(returns, cov)
    n = len(S)
    if cap * n < 1 - 1e-12:
        raise ValueError("Infeasible constraint: cap times the number of assets is below one")
    s = np.sqrt(np.maximum(np.diag(S), 1e-14))
    w = 1.0 / s
    w = w / w.sum()
    if cap < 1:
        w = _proj(w, cap)
    return pd.Series(w, names) if names is not None else w
