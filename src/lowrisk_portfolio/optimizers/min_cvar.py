"""Historical minimum CVaR via the Rockafellar-Uryasev linear program. Solver failure raises an error.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def cvar(w, returns, alpha=0.10):
    "Historical portfolio CVaR; positive values denote losses, matching the LP objective."
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be a tail probability in (0, 1]")
    loss = -np.asarray(returns, float) @ np.asarray(w, float)
    if loss.size == 0 or not np.isfinite(loss).all():
        raise ValueError("Loss scenarios must be finite")
    T = len(loss)
    z = np.sort(loss)[::-1]
    k = alpha * T
    m = int(np.floor(k))
    return float((z[:m].sum() + (k - m) * (z[m] if m < T else 0.0)) / k)


def min_cvar(returns, cap=1.0, alpha=0.10):
    names = returns.columns if isinstance(returns, pd.DataFrame) else None
    X = np.asarray(returns, float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("returns must be a nonempty, finite two-dimensional array")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be a tail probability in (0, 1]")
    T, n = X.shape
    if cap * n < 1 - 1e-12:
        raise ValueError("Infeasible constraint: cap times the number of assets is below one")
    c = np.concatenate([np.zeros(n), [1.0], np.ones(T) / (alpha * T)])
    A = np.hstack([-X, -np.ones((T, 1)), -np.eye(T)])  # -r_t'w - ζ - z_t ≤ 0
    Aeq = np.zeros((1, n + 1 + T))
    Aeq[0, :n] = 1.0
    bounds = [(0.0, min(cap, 1.0))] * n + [(None, None)] + [(0.0, None)] * T
    r = linprog(c, A_ub=A, b_ub=np.zeros(T), A_eq=Aeq, b_eq=[1.0], bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError("CVaR optimization failed: " + r.message)
    w = np.maximum(r.x[:n], 0.0)
    ssum = w.sum()
    if not np.isfinite(w).all() or ssum <= 0:
        raise RuntimeError("CVaR optimization returned invalid weights")
    w = w / ssum
    return pd.Series(w, names) if names is not None else w
