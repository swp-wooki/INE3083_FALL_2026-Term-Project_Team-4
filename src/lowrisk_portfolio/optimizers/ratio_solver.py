"""Constrained ratio optimization, including certified positive- and negative-numerator branches.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np
from scipy.optimize import lsq_linear, minimize


def checked_covariance(cov):
    Q = np.array(cov, dtype=float, order="C", copy=True)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or not len(Q):
        raise ValueError("cov must be a nonempty square matrix")
    if not np.isfinite(Q).all() or not np.allclose(Q, Q.T, rtol=1e-11, atol=1e-14):
        raise ValueError("cov must be finite and symmetric")
    Q = (Q + Q.T) / 2
    if np.linalg.eigvalsh(Q)[0] <= 0:
        raise ValueError("The certified ratio solver requires positive definite covariance")
    return Q


def linear_maximum(a, cap):
    """Exact linear oracle; stable ordering fixes ties deterministically."""
    w = np.zeros(len(a))
    remaining = 1.0
    for j in np.argsort(-a, kind="stable"):
        w[j] = min(cap, remaining)
        remaining -= w[j]
        if remaining <= 0:
            break
    return w


def _positive_ratio(Q, a, cap, start, maxiter, tol):
    n = len(a)
    A = cap * np.ones((n, n)) - np.eye(n)
    y0 = start / (a @ start)
    result = minimize(
        lambda y: y @ Q @ y,
        y0,
        jac=lambda y: 2 * Q @ y,
        bounds=[(0, None)] * n,
        constraints=[
            {"type": "eq", "fun": lambda y: a @ y - 1, "jac": lambda y: a},
            {"type": "ineq", "fun": lambda y: A @ y, "jac": lambda y: A},
        ],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": maxiter},
    )
    y = result.x
    # A linear KKT solve removes SLSQP's final active-set/roundoff error.
    active_tol = 1e-7 * max(1.0, np.max(np.abs(y)))
    C = np.vstack([A[A @ y < active_tol], np.eye(n)[y < active_tol]])
    E = np.vstack([a, -C])
    K = np.block([[2 * Q, E.T], [E, np.zeros((len(E), len(E)))]])
    rhs = np.r_[np.zeros(n), 1.0, np.zeros(len(C))]
    solution = np.linalg.lstsq(K, rhs, rcond=None)[0]
    candidate, dual = solution[:n], solution[n:]
    if (
        np.max(np.abs(K @ solution - rhs)) < 1e-8
        and abs(a @ candidate - 1) < 1e-9
        and candidate.min() >= -1e-10
        and (A @ candidate).min() >= -1e-10
        and (not len(C) or dual[1:].min() >= -1e-10)
    ):
        y = candidate
    # Nonnegative inequality multipliers yield a valid dual lower bound even
    # when stationarity is only approximate.
    C = np.vstack([A[A @ y < active_tol], np.eye(n)[y < active_tol]])
    B = np.column_stack([a, -C.T])
    fit = lsq_linear(
        B,
        -2 * Q @ y,
        bounds=(np.r_[-np.inf, np.zeros(len(C))], np.full(1 + len(C), np.inf)),
        tol=1e-13,
        max_iter=1000,
    )
    h = B @ fit.x
    primal = float(y @ Q @ y)
    lower = float(-0.25 * h @ np.linalg.solve(Q, h) - fit.x[0])
    gap = max(0.0, primal - lower) / max(primal, 1e-30)
    residual = max(abs(a @ y - 1), max(0.0, -float(y.min())), max(0.0, -float((A @ y).min())))
    if not np.isfinite(gap) or gap > tol or residual > 1e-8:
        raise RuntimeError(
            "Ratio QP failed certification: "
            f"relative duality gap={gap:.3g}, feasibility={residual:.3g}; "
            f"{result.message}"
        )
    w = y / y.sum()
    # Only eliminate floating-point negative zeros, never project a solution.
    w[np.abs(w) < 1e-14] = 0.0
    w /= w.sum()
    return w, {
        "method": "homogeneous_convex_qp",
        "relative_duality_gap": gap,
        "primal_residual": residual,
        "iterations": int(result.nit),
    }


def _negative_ratio(Q, a, cap, tol, max_nodes=2_000_000):
    """Search all capped-simplex vertices, pruning only with an objective bound.

    A vertex has k=floor(1/cap) entries cap and at most one residual entry r.
    For fixed residual coordinate, divide all weights by cap and choose k
    unit entries. If A is selected and U remains, completion B (|B|=k) has
    q=q_A+sum_j[2(Q1_A)_j+Q_jj]+sum_{i!=j in B}Q_ij.
    Replace each row's last sum by its largest k-1 entries in U to get
    q <= q_A+sum_{j in B}d_j. With b=-a and any v,
    (b_A+sum b_j)^2 >= 2v(b_A+sum b_j)-v^2.
    Therefore min_B (b'z)^2-t^2 q is bounded below by choosing the k smallest
    coefficients 2v*b_j-t^2*d_j. A nonnegative lower bound excludes any
    completion with -Sharpe<t. Use t=(1-tol)*incumbent and a floating-point
    error margin. Every pruned node then respects the relative objective
    tolerance. Greedy swaps provide only an incumbent, never a certificate.
    """
    n = len(a)
    reciprocal = 1.0 / cap
    nearest = round(reciprocal)
    if abs(reciprocal - nearest) < 1e-12:
        k, residual = nearest, 0.0
    else:
        k = int(np.floor(reciprocal))
        residual = 1.0 - k * cap
    best, best_w = np.inf, None
    visited, leaves, pruned = 0, 0, 0
    eps = np.finfo(float).eps
    b_original = -a
    pivots = [None] if residual == 0 else range(n)
    for pivot in pivots:
        ids = np.array([j for j in range(n) if j != pivot], dtype=int)
        ids = ids[np.argsort(b_original[ids], kind="stable")]
        b = b_original[ids]
        S = Q[np.ix_(ids, ids)]
        size = len(ids)
        r = residual / cap
        base_b = 0.0 if pivot is None else r * b_original[pivot]
        base_q = 0.0 if pivot is None else r * r * Q[pivot, pivot]
        base_rows = np.zeros(size) if pivot is None else r * Q[ids, pivot]

        def exact(c):
            bb = base_b + b[c].sum()
            qq = base_q + 2 * base_rows[c].sum() + S[np.ix_(c, c)].sum()
            return float(bb / np.sqrt(qq))

        def accept(c, value):
            nonlocal best, best_w
            if value < best:
                best = value
                best_w = np.zeros(n)
                best_w[ids[c]] = cap
                if pivot is not None:
                    best_w[pivot] = residual

        chosen = list(range(k))
        seed = exact(chosen)
        # Deterministic one-swap local improvement, followed by global search.
        while True:
            improved = False
            for position in range(k):
                for j in range(size):
                    if j in chosen:
                        continue
                    candidate = chosen.copy()
                    candidate[position] = j
                    value = exact(candidate)
                    if value < seed - 16 * eps * max(1.0, abs(seed)):
                        chosen, seed, improved = candidate, value, True
            if not improved:
                break
        accept(chosen, seed)
        row_bounds = {}
        for j in range(size):
            block = S[j:, j:].copy()
            np.fill_diagonal(block, -np.inf)
            ordered = np.sort(block, axis=1)[:, ::-1]
            sums = np.cumsum(ordered[:, : max(0, min(k - 1, size - j - 1))], axis=1)
            for remaining in range(1, min(k, size - j) + 1):
                row_bounds[j, remaining] = S.diagonal()[j:] + (
                    sums[:, remaining - 2] if remaining > 1 else 0
                )

        def visit(j, remaining, selected, bs, qs, rows):
            nonlocal visited, leaves, pruned
            visited += 1
            if visited > max_nodes:
                raise RuntimeError(
                    "Negative-ratio global search exceeded node limit; "
                    "no uncertified portfolio was returned"
                )
            if remaining == 0 or size - j == remaining:
                c = selected + list(range(j, size)) if remaining else selected
                accept(c, exact(c))
                leaves += 1
                return
            d = 2 * rows[j:] + row_bounds[j, remaining]
            target2 = (best * (1.0 - tol)) ** 2
            bmin = bs + b[j : j + remaining].sum()
            bmax = bs + b[size - remaining :].sum()
            # Each tangent is independently valid; these three tight, cheap
            # choices avoid an additional numerical optimizer inside the tree.
            for v in (bmin, (bmin + bmax) / 2, bmax):
                coefficients = 2 * v * b[j:] - target2 * d
                smallest = np.partition(coefficients, remaining - 1)[:remaining]
                lower = 2 * v * bs - v * v - target2 * qs + smallest.sum()
                magnitude = abs(2 * v * bs) + v * v + abs(target2 * qs) + np.abs(smallest).sum()
                margin = 256 * eps * magnitude
                if lower >= margin:
                    pruned += 1
                    return
            visit(
                j + 1,
                remaining - 1,
                selected + [j],
                bs + b[j],
                qs + 2 * rows[j] + S[j, j],
                rows + S[:, j],
            )
            visit(j + 1, remaining, selected, bs, qs, rows)

        visit(0, k, [], base_b, base_q, base_rows)
    best_w /= best_w.sum()
    return best_w, {
        "method": "vertex_branch_and_bound",
        "nodes": visited,
        "evaluated_leaves": leaves,
        "pruned_nodes": pruned,
        "relative_objective_tolerance": tol,
    }


def maximize_ratio(cov, numerator, cap=1.0, maxiter=1000, return_info=False):
    """Maximize numerator'w/sqrt(w'cov w), or raise if certification fails."""
    S = checked_covariance(cov)
    a = np.array(numerator, dtype=float, order="C", copy=True)
    n = len(S)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError("numerator must be a finite vector matching cov")
    if not np.isfinite(cap) or cap <= 0 or cap * n < 1 - 1e-12:
        raise ValueError("cap must define a nonempty capped simplex")
    cap = min(float(cap), 1.0)
    if abs(cap * n - 1) < 1e-12:
        w, info = np.full(n, 1 / n), {"method": "singleton_feasible_set"}
    elif not np.any(a):
        w, info = np.full(n, 1 / n), {"method": "zero_numerator"}
    else:
        Q, a = S / (np.trace(S) / n), a / np.max(np.abs(a))
        start = linear_maximum(a, cap)
        maximum = float(a @ start)
        if maximum > 0:
            w, info = _positive_ratio(Q, a, cap, start, maxiter, 1e-8)
        elif maximum == 0:
            w, info = start, {"method": "zero_maximum_numerator"}
        else:
            w, info = _negative_ratio(Q, a, cap, 1e-10)
    if (
        not np.isfinite(w).all()
        or abs(w.sum() - 1) > 1e-9
        or w.min() < -1e-10
        or w.max() > cap + 1e-9
    ):
        raise RuntimeError("Certified ratio solution violates portfolio constraints")
    return (w, info) if return_info else w
