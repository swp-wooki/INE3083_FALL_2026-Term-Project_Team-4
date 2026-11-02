"""Fixed mixture (1-a)/N + a*w_GMVP. The experiment uses a=0.5, not an estimated Tu-Zhou coefficient.

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


def _gmvp(S, cap, iters=5000, tol=1e-10):
    n = len(S)
    L = 2 * np.linalg.eigvalsh(S)[-1] * 1.0001 + 1e-18
    w = _proj(np.full(n, 1.0 / n), cap)
    y = w.copy()
    tk = 1.0
    for _ in range(iters):
        wn = _proj(y - 2 * S @ y / L, cap)
        if np.abs(wn - w).max() < tol:
            w = wn
            break
        if (y - wn) @ (wn - w) > 0:
            tk = 1.0
            y = wn
        else:
            tn = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
            y = wn + (tk - 1) / tn * (wn - w)
            tk = tn
        w = wn
    return w


def _cap_of(cap, n):
    c = float(cap(n)) if callable(cap) else float(cap)
    if c * n < 1 - 1e-12:
        raise ValueError("Infeasible constraint: cap times n is below one (n=%d, cap=%g)" % (n, c))
    return c


def mix_1n(returns=None, cov=None, a=0.5, cap=1.0):
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
    n = len(S)
    c = _cap_of(cap, n)
    w = (1 - a) * np.full(n, 1.0 / n) + a * _gmvp(S, c)
    return pd.Series(w, names) if names is not None else w
