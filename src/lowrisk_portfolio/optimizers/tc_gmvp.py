"""Transaction-cost GMVP with an L1 penalty on drifted weight changes. Pass cost=2*c for one-way trading rate c.

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


def _prox(v, cap, lam, a):
    "Proximal map for 0.5*||w-v||^2 + lam*||w-a||_1 on the capped simplex; lam=0 gives projection."

    def w_of(th):
        z = v - th - a
        return np.clip(a + np.sign(z) * np.maximum(np.abs(z) - lam, 0), 0, cap)

    def slope(th):
        z = v - th - a
        u = a + np.sign(z) * np.maximum(np.abs(z) - lam, 0)
        return float(((np.abs(z) > lam) & (u > 0) & (u < cap)).sum())

    th = _theta(
        lambda th: w_of(th).sum() - 1,
        slope,
        v.min() - lam - cap - 2.0,
        v.max() + lam + 1.0,
    )
    return w_of(th)


def _qp(S, cap, lam=0.0, a=None, w0=None, iters=5000, tol=1e-10):
    """min w'Sw + lam·‖w-a‖₁  s.t. Σw = 1, 0 ≤ w ≤ cap"""
    n = len(S)
    a = np.zeros(n) if a is None else a
    L = 2 * np.linalg.eigvalsh(S)[-1] * 1.0001 + 1e-18
    pr = lambda v: _prox(v, cap, lam / L, a)
    w = pr(np.full(n, 1.0 / n) if w0 is None else np.asarray(w0, float))
    y = w.copy()
    tk = 1.0
    for _ in range(iters):
        wn = pr(y - 2 * S @ y / L)
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


def tc_gmvp(cov, w_prev=None, cost=0.002, gamma=5.0, cap=1.0):
    names = cov.columns if isinstance(cov, pd.DataFrame) else None
    S = np.asarray(cov, float)
    n = len(S)
    c = _cap_of(cap, n)
    if w_prev is None:
        a = np.zeros(n)
    elif isinstance(w_prev, pd.Series) and names is not None:
        a = w_prev.reindex(names).fillna(0.0).values
    else:
        a = np.asarray(w_prev, float)
    if a.min() < 0 or a.sum() > 1 + 1e-9:
        raise ValueError("w_prev must be nonnegative and sum to at most one")
    w = _qp(S, c, cost / gamma, a, w0=a if a.sum() > 0 else None)
    return pd.Series(w, names) if names is not None else w
