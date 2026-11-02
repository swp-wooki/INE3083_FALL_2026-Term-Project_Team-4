"""Minimum variance with a squared-weight norm constraint and position limits.

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


def _gmvp(S, cap, w0=None, iters=5000, tol=1e-10):
    n = len(S)
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


def _norm_solve(S, cap, delta, w_start=None, x_lo=-8.0, rtol=2e-4):
    "Return (w, x), where x = log10(nu / mean variance). Reuse x as x_lo when tightening delta."
    n = len(S)
    I = np.eye(n)
    sc = np.trace(S) / n
    # At the smallest feasible norm the only portfolio is equal weight.
    if delta <= 1.0 / n + 1e-14:
        return np.full(n, 1.0 / n), x_lo
    # An inactive constraint has multiplier zero, not a small positive ridge.
    w_base = _gmvp(S, cap, w_start)
    if w_base @ w_base <= delta:
        return w_base, x_lo
    sol = lambda x, w0: _gmvp(S + 10**x * sc * I, cap, w0)
    g = lambda w: np.log(w @ w) - np.log(delta)
    w_lo = sol(x_lo, w_start)
    g_lo = g(w_lo)
    if g_lo <= 0:
        return w_lo, x_lo
    x_hi = x_lo
    while True:
        x_hi += 2.0
        w_hi = sol(x_hi, w_lo)
        g_hi = g(w_hi)
        if g_hi <= 0 or x_hi > 6:
            break
        x_lo, w_lo, g_lo = x_hi, w_hi, g_hi
    side = 0
    gt_hi = g_hi
    for _ in range(60):
        if gt_hi >= np.log1p(-rtol) or x_hi - x_lo < 1e-6:
            break
        x = (x_lo * g_hi - x_hi * g_lo) / (g_hi - g_lo)
        w = sol(x, w_hi)
        gx = g(w)
        if gx > 0:
            x_lo, w_lo, g_lo = x, w, gx
            if side == -1:
                g_hi /= 2
            side = -1
        else:
            x_hi, w_hi, g_hi, gt_hi = x, w, gx, gx
            if side == 1:
                g_lo /= 2
            side = 1
    return w_hi, x_hi


def norm_gmvp(returns=None, cov=None, delta=None, min_eff=None, cap=1.0, rtol=2e-4):
    "Specify delta (upper bound on squared norm) or min_eff (minimum effective asset count)."
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
    if (delta is None) == (min_eff is None):
        raise ValueError("Specify exactly one of delta and min_eff")
    delta = 1.0 / min_eff if delta is None else float(delta)
    if delta < 1.0 / n - 1e-12:
        raise ValueError("Infeasible delta < 1/n: the effective asset count cannot exceed n")
    w, _ = _norm_solve(S, c, delta, rtol=rtol)
    return pd.Series(w, names) if names is not None else w
