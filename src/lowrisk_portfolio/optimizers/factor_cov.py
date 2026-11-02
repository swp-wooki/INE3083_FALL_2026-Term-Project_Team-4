"""Factor covariance B Cov(f) B.T + diagonal residual variance, and capped long-only GMVP.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np
import pandas as pd


def factor_cov(returns, factors=None, scale=1.0, min_obs=None):
    R = pd.DataFrame(returns).astype(float)
    if factors is None:
        F = R.mean(axis=1).to_frame("EW")
    else:
        F = pd.DataFrame(factors).astype(float).reindex(R.index)
    ok_f = F.notna().all(axis=1)
    R, F = R[ok_f], F[ok_f]
    T, K = F.shape
    n = R.shape[1]
    min_obs = max(K + 10, int(0.75 * T)) if min_obs is None else min_obs
    Z = np.column_stack([np.ones(T), F.values])
    B = np.zeros((n, K))
    se2 = np.zeros(n)
    bad = []
    for j, c in enumerate(R.columns):
        m = R[c].notna().values
        if m.sum() < min_obs:
            bad.append(c)
            continue
        y = R[c].values[m]
        Zj = Z[m]
        coef, *_ = np.linalg.lstsq(Zj, y, rcond=None)
        e = y - Zj @ coef
        B[j] = coef[1:]
        se2[j] = e @ e / (m.sum() - K - 1)
    if bad:
        raise ValueError(
            "Assets with fewer than min_obs=%d observations: %s" % (min_obs, list(bad)[:10])
        )
    Sf = np.atleast_2d(np.cov(F.values, rowvar=False))
    S = scale * (B @ Sf @ B.T + np.diag(se2))
    return pd.DataFrame(S, R.columns, R.columns)


def _theta(fun, slope, lo, hi, tol=1e-13):
    "Find a root of decreasing piecewise-linear f(theta) with safeguarded Newton and bisection."
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


def gmvp(cov, cap=1.0, w0=None, iters=5000, tol=1e-10):
    names = cov.columns if isinstance(cov, pd.DataFrame) else None
    S = np.asarray(cov, float)
    n = len(S)
    if cap * n < 1 - 1e-12:
        raise ValueError("Infeasible constraint: cap times the number of assets is below one")
    L = 2 * np.linalg.eigvalsh(S)[-1] * 1.0001 + 1e-18
    w = _proj(np.full(n, 1.0 / n) if w0 is None else np.asarray(w0, float), cap)
    y = w.copy()
    tk = 1.0
    for _ in range(iters):
        wn = _proj(y - 2 * S @ y / L, cap)
        if np.abs(wn - w).max() < tol:
            w = wn
            break
        if (y - wn) @ (wn - w) > 0:
            tk = 1.0
            y = wn  # Adaptive restart
        else:
            tn = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
            y = wn + (tk - 1) / tn * (wn - w)
            tk = tn
        w = wn
    return pd.Series(w, names) if names is not None else w
