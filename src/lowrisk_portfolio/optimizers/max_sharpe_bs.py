"""Bayes-Stein expected returns and constrained maximum-Sharpe allocation.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np
import pandas as pd

from .ratio_solver import maximize_ratio


def bayes_stein(X, S=None):
    X = np.array(X, dtype=float, order="C", copy=True)
    T, n = X.shape
    if not np.isfinite(X).all():
        raise ValueError("returns must be finite")
    mu = X.mean(axis=0)
    if T > n + 2:
        Sig = np.cov(X, rowvar=False) * (T - 1) / (T - n - 2)
    elif S is None:
        raise ValueError(
            "T=%d <= number of assets + 2=%d: Bayes-Stein requires a shrinkage covariance supplied through cov."
            % (T, n + 2)
        )
    else:
        Sig = np.asarray(S, float)
    ev = np.linalg.eigvalsh(Sig)
    if ev[0] <= 1e-10 * ev[-1]:
        raise ValueError(
            "Bayes-Stein covariance is not positive definite (min/max eigenvalue %.1e)"
            % (ev[0] / ev[-1])
        )
    Si = np.linalg.inv(Sig)
    one = np.ones(n)
    mu0 = float(one @ Si @ mu) / float(one @ Si @ one)
    diff = mu - mu0
    q = float(diff @ Si @ diff)
    ph = (n + 2) / ((n + 2) + T * q)
    return (1 - ph) * mu + ph * mu0


def max_sharpe_bs(returns, cov=None, cap=1.0, rf=0.0, iters=1000):
    names = returns.columns if isinstance(returns, pd.DataFrame) else None
    X = np.array(returns, dtype=float, order="C", copy=True)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("returns must be a finite two-dimensional array")
    if not np.isfinite(rf):
        raise ValueError("rf must be finite")
    S = np.cov(X, rowvar=False) if cov is None else np.asarray(cov, float)
    m = bayes_stein(X, cov) - rf
    w = maximize_ratio(S, m, cap=cap, maxiter=iters)
    return pd.Series(w, names) if names is not None else w
