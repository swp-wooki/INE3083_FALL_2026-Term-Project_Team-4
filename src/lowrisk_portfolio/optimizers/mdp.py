"""Maximum diversification using the certified constrained ratio solver.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np
import pandas as pd

from .ratio_solver import maximize_ratio


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


def diversification_ratio(w, cov):
    w = np.asarray(w, float)
    S = np.asarray(cov, float)
    return float(np.sqrt(np.maximum(np.diag(S), 1e-14)) @ w / np.sqrt(w @ S @ w))


def mdp(returns=None, cov=None, cap=1.0, iters=1000):
    S, names = _inputs(returns, cov)
    w = maximize_ratio(S, np.sqrt(np.diag(S)), cap=cap, maxiter=iters)
    return pd.Series(w, names) if names is not None else w
