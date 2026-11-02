"""Variance timing: weights proportional to variance**(-eta), projected onto the capped simplex. The experiment uses eta=1.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np
import pandas as pd


def _theta(fun, slope, lo, hi, tol=1e-13):
    th = 0.5 * (lo + hi)
    for _ in range(200):
        f = fun(th)
        if abs(f) < tol:
            break
        if f > 0:
            lo = th
        else:
            hi = th
        k = slope(th)
        nt = th + f / k if k > 0 else None
        th = nt if (nt is not None and lo < nt < hi) else 0.5 * (lo + hi)
        if hi - lo < 1e-16:
            break
    return th


def _proj(v, cap):
    "Euclidean projection onto the capped simplex: sum(w) = 1 and 0 <= w <= cap."
    fun = lambda th: np.clip(v - th, 0, cap).sum() - 1
    slope = lambda th: float(((v - th > 0) & (v - th < cap)).sum())
    return np.clip(v - _theta(fun, slope, v.min() - 1.0, v.max()), 0, cap)


def vol_timing(returns=None, cov=None, variances=None, eta=1.0, cap=1.0):
    names = None
    if variances is not None:
        if isinstance(variances, pd.Series):
            names = variances.index
        v = np.asarray(variances, float)
    elif cov is not None:
        if isinstance(cov, pd.DataFrame):
            names = cov.columns
        v = np.diag(np.asarray(cov, float))
    elif returns is not None:
        if isinstance(returns, pd.DataFrame):
            names = returns.columns
        X = np.asarray(returns, float)
        if np.isnan(X).any():
            raise ValueError("returns contains missing values")
        v = X.var(axis=0, ddof=1)
    else:
        raise ValueError("Specify returns, cov, or variances")
    if np.isnan(v).any():
        raise ValueError("Variances contain missing values")
    n = len(v)
    if cap * n < 1 - 1e-12:
        raise ValueError("Infeasible constraint: cap times the number of assets is below one")
    lw = -eta * np.log(np.maximum(v, 1e-300))
    w = np.exp(lw - lw.max())
    w = w / w.sum()  # Log scaling prevents overflow for large eta
    if cap < 1:
        w = _proj(w, cap)
    return pd.Series(w, names) if names is not None else w
