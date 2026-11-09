"""Constrained minimum-variance problems absent from the supplied optimizers.

Each solver returns weights and a duality-based bound on the relative objective error, so
a failed solve raises instead of silently passing an inaccurate portfolio to the backtest.
"""

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

from lowrisk_portfolio.models import module

GAP_LIMIT = 1e-6
FEASIBILITY_LIMIT = 1e-8


def _scaled(S):
    scale = float(np.trace(S) / len(S))
    return S / scale, scale


def _polish_equalities(w, A, c, cap):
    """Remove rounding residuals from Aw=c while respecting the bounds."""
    w = np.clip(w, 0, cap)
    for _ in range(50):
        residual = c - A @ w
        if np.abs(residual).max() < 1e-13:
            break
        w = np.clip(w + np.linalg.lstsq(A, residual, rcond=None)[0], 0, cap)
    return w


def exposure_neutral_gmvp(S, A, c, cap, start):
    """min w'Sw subject to Aw=c and 0<=w<=cap; A's first row is the budget."""
    Ss, _ = _scaled(S)

    def certify(w):
        # f is convex, so min over the polytope of the linearization bounds f(w)-f*.
        g = 2 * Ss @ w
        lp = linprog(g, A_eq=A, b_eq=c, bounds=[(0, cap)] * len(w), method="highs")
        if not lp.success:
            raise RuntimeError("Exposure-neutral certificate LP failed: " + lp.message)
        return float(max(g @ w - lp.fun, 0) / (w @ Ss @ w))

    def solve(method, w0):
        constraint = dict(type="eq", fun=lambda w: A @ w - c, jac=lambda w: A)
        options = dict(ftol=1e-15, maxiter=3000) if method == "SLSQP" else dict(
            gtol=1e-10, xtol=1e-12, maxiter=20000)
        kwargs = dict(hess=lambda w: 2 * Ss) if method == "trust-constr" else {}
        result = minimize(lambda w: w @ Ss @ w, w0, jac=lambda w: 2 * Ss @ w, method=method,
                          bounds=[(0, cap)] * len(w0), constraints=[constraint], options=options, **kwargs)
        return _polish_equalities(result.x, A, c, cap)

    w = solve("SLSQP", start)
    gap = certify(w)
    if gap > GAP_LIMIT:
        w = solve("trust-constr", w)
        gap = certify(w)
    residual = float(np.abs(A @ w - c).max())
    if gap > GAP_LIMIT or residual > FEASIBILITY_LIMIT:
        raise RuntimeError(f"Exposure-neutral solve failed: gap={gap}, residual={residual}")
    return w, dict(relative_gap=gap, max_constraint_residual=residual)


def _lagrangian_gap(Ss, b, tau, w, mu, cap):
    """Weak-duality bound on (f(w)-f*)/f(w) for any mu>=0 and feasible w."""
    d = w - b
    g = 2 * Ss @ w + mu * 2 * Ss @ d
    remaining, best = 1.0, 0.0
    for i in np.argsort(g):
        take = min(cap, remaining)
        best += take * g[i]
        remaining -= take
        if remaining <= 0:
            break
    f = w @ Ss @ w
    dual = f + mu * (d @ Ss @ d - tau) - (g @ w - best)
    return float(max(f - dual, 0) / f)


def _capped_linear_min(g, cap):
    remaining, best = 1.0, 0.0
    for i in np.argsort(g):
        take = min(cap, remaining)
        best += take * g[i]
        remaining -= take
        if remaining <= 0:
            break
    return best


def effective_n_gmvp(S, min_effective, cap, rtol=1e-6):
    """Supplied norm-constrained GMVP with a weak-duality accuracy bound on w'w<=1/min_effective."""
    Ss, _ = _scaled(S)
    delta = 1 / min_effective
    w = module("norm_gmvp").norm_gmvp(cov=S, min_eff=min_effective, cap=cap, rtol=rtol)
    if w @ w > delta * (1 + 1e-9):
        raise RuntimeError("Effective-holdings constraint violated")

    def gap_at(nu):
        g = 2 * Ss @ w + 2 * nu * w
        f = w @ Ss @ w
        dual = f + nu * (w @ w - delta) - (g @ w - _capped_linear_min(g, cap))
        return float(max(f - dual, 0) / f)

    free = (w > 1e-7) & (w < cap - 1e-7)
    x, y = (2 * Ss @ w)[free], (2 * w)[free]
    x, y = x - x.mean(), y - y.mean()
    nu = max(-float(x @ y) / float(y @ y), 0.0) if y @ y > 0 else 0.0
    binding = bool(w @ w > delta * (1 - 1e-4))
    gap = gap_at(nu if binding else 0.0)
    if gap > 1e-5:
        scan = minimize_scalar(gap_at, bounds=(0.0, 4 * nu + 1), method="bounded", options=dict(xatol=1e-10))
        gap = min(gap, float(scan.fun))
    if gap > 1e-5:
        raise RuntimeError(f"Effective-holdings solve failed: gap={gap}")
    return w, dict(relative_gap=gap, binding=binding)


def sector_capped_gmvp(S, sectors, sector_cap, cap):
    """min w'Sw with a cap on every sector's total weight, plus budget and position limits."""
    Ss, _ = _scaled(S)
    labels = np.unique(sectors)
    G = np.array([(sectors == label).astype(float) for label in labels])
    limit = np.full(len(labels), sector_cap)
    budget = np.ones((1, len(S)))
    gmvp = module("factor_cov").gmvp(S, cap=cap)

    def certify(w):
        g = 2 * Ss @ w
        lp = linprog(g, A_ub=G, b_ub=limit, A_eq=budget, b_eq=np.ones(1), bounds=[(0, cap)] * len(w),
                     method="highs")
        if not lp.success:
            raise RuntimeError("Sector-cap certificate LP failed: " + lp.message)
        return float(max(g @ w - lp.fun, 0) / (w @ Ss @ w))

    if (G @ gmvp).max() <= sector_cap:
        gap = certify(gmvp)
        if gap > GAP_LIMIT:
            raise RuntimeError(f"Unconstrained GMVP inaccurate: gap={gap}")
        return gmvp, dict(relative_gap=gap, binding=False)
    constraints = [dict(type="eq", fun=lambda w: w.sum() - 1, jac=lambda w: np.ones_like(w)),
                   dict(type="ineq", fun=lambda w: limit - G @ w, jac=lambda w: -G)]
    result = minimize(lambda w: w @ Ss @ w, gmvp, jac=lambda w: 2 * Ss @ w, method="SLSQP",
                      bounds=[(0, cap)] * len(gmvp), constraints=constraints,
                      options=dict(ftol=1e-15, maxiter=3000))
    w = _polish_equalities(result.x, budget, np.ones(1), cap)
    excess = float((G @ w - limit).max())
    gap = certify(w)
    if gap > GAP_LIMIT or excess > FEASIBILITY_LIMIT:
        raise RuntimeError(f"Sector-cap solve failed: gap={gap}, excess={excess}")
    return w, dict(relative_gap=gap, binding=True)


def te_limited_gmvp(S, b, te_annual, cap):
    """min w'Sw subject to 12(w-b)'S(w-b)<=te^2, budget, and 0<=w<=cap."""
    Ss, scale = _scaled(S)
    tau = te_annual**2 / 12 / scale
    gmvp = module("factor_cov").gmvp(S, cap=cap)
    d = gmvp - b
    te_gmvp2 = float(d @ Ss @ d)
    if te_gmvp2 <= tau:
        gap = _lagrangian_gap(Ss, b, tau, gmvp, 0.0, cap)
        if gap > GAP_LIMIT:
            raise RuntimeError(f"Unconstrained GMVP inaccurate: gap={gap}")
        return gmvp, dict(relative_gap=gap, binding=False, multiplier=0.0,
                          ex_ante_te=float(np.sqrt(12 * d @ S @ d)))
    # Start on the segment from b to the GMVP, strictly inside the tracking-error limit.
    start = b + 0.9 * np.sqrt(tau / te_gmvp2) * d
    budget = dict(type="eq", fun=lambda w: w.sum() - 1, jac=lambda w: np.ones_like(w))
    limit = dict(type="ineq", fun=lambda w: 1 - (w - b) @ Ss @ (w - b) / tau,
                 jac=lambda w: -2 * Ss @ (w - b) / tau)
    result = minimize(lambda w: w @ Ss @ w, start, jac=lambda w: 2 * Ss @ w, method="SLSQP",
                      bounds=[(0, cap)] * len(b), constraints=[budget, limit],
                      options=dict(ftol=1e-15, maxiter=3000))
    A = np.ones((1, len(b)))
    w = _polish_equalities(result.x, A, np.ones(1), cap)
    # Restore strict feasibility by moving toward b, which stays in the convex feasible set.
    excess = (w - b) @ Ss @ (w - b) / tau
    if excess > 1:
        w = b + (w - b) / np.sqrt(excess) * (1 - 1e-12)

    def gap_at(mu):
        return _lagrangian_gap(Ss, b, tau, w, mu, cap)

    free = (w > 1e-7) & (w < cap - 1e-7)
    x = (2 * Ss @ w)[free]
    y = (2 * Ss @ (w - b))[free]
    x, y = x - x.mean(), y - y.mean()
    mu = max(-float(x @ y) / float(y @ y), 0.0) if y @ y > 0 else 0.0
    gap = gap_at(mu)
    if gap > GAP_LIMIT:
        scan = minimize_scalar(gap_at, bounds=(0.0, 4 * mu + 1), method="bounded",
                               options=dict(xatol=1e-10))
        if scan.fun < gap:
            mu, gap = float(scan.x), float(scan.fun)
    if gap > GAP_LIMIT:
        raise RuntimeError(f"Tracking-error solve failed: gap={gap}")
    return w, dict(relative_gap=gap, binding=True, multiplier=mu,
                   ex_ante_te=float(np.sqrt(12 * (w - b) @ S @ (w - b))))
