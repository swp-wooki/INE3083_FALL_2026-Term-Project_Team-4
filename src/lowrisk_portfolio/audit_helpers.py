"""Independent numerical references for the final observed-return experiment."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, linprog

from .config import Config
from .data import load_panel
from .models import module as module
from .paths import OUTPUT as OUT
from .paths import ROOT

CAP = 2 / 30


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def projection(v, cap):
    th = brentq(lambda x: np.clip(v - x, 0, cap).sum() - 1, v.min() - 1, v.max(), xtol=1e-14)
    return np.clip(v - th, 0, cap)


def linear_min(g, cap):
    w = np.zeros(len(g))
    left = 1.0
    for i in np.argsort(g):
        w[i] = min(cap, left)
        left -= w[i]
        if left <= 1e-14:
            break
    return w


def gap(S, w, cap=CAP):
    g = 2 * S @ w
    return max(0.0, float(g @ (w - linear_min(g, cap)))) / max(float(w @ S @ w), 1e-30)


def lwcc_ref(X, ddof=0):
    # Direct moment expectations, independently assembled from outer products.
    Y = X - X.mean(0)
    n = len(Y) - ddof
    p = X.shape[1]
    outer = np.einsum("ti,tj->tij", Y, Y)
    S = outer.sum(0) / n
    sd = np.sqrt(S.diagonal())
    rho = np.corrcoef(X, rowvar=False)[np.triu_indices(p, 1)].mean()
    F = rho * np.outer(sd, sd)
    np.fill_diagonal(F, S.diagonal())
    pi = (outer**2).sum(0) / n - S * S
    cross = np.einsum("ti,tij->ij", Y * Y, outer) / n - S.diagonal()[:, None] * S
    off = ~np.eye(p, dtype=bool)
    rh = np.trace(pi) + rho * np.sum((sd[None, :] / sd[:, None] * cross)[off])
    den = np.sum((S - F) ** 2)
    a = np.clip((pi.sum() - rh) / den / n, 0, 1) if den else 1.0
    return a * F + (1 - a) * S


def qis_ref(X):
    # Scalar-per-eigenvalue evaluation of the author's 01/2024 formula.
    Y = X - X.mean(0)
    p = X.shape[1]
    n = len(X) - 1
    c = p / n
    eig, U = np.linalg.eigh(Y.T @ Y / n)
    inv = 1 / eig[max(0, p - n) :]
    h = min(c * c, 1 / (c * c)) ** 0.35 / p**0.35
    theta = []
    H = []
    for u in inv:
        diff = inv - u
        den = diff * diff + (h * inv) ** 2
        theta.append(np.mean(inv * diff / den))
        H.append(np.mean(h * inv * inv / den))
    a2 = np.square(theta) + np.square(H)
    if p <= n:
        d = 1 / (inv * ((1 - c) ** 2 + 2 * c * (1 - c) * np.array(theta) + c * c * a2))
    else:
        d = np.r_[np.repeat(1 / ((c - 1) * inv.mean()), p - n), 1 / (inv * a2)]
    d *= eig.sum() / d.sum()
    return (U * d) @ U.T


def factor_ref(X, f):
    z = np.column_stack([np.ones(len(f)), f.drop(columns="RF")])
    y = X - f.RF.to_numpy()[:, None]
    co = np.linalg.lstsq(z, y, rcond=None)[0]
    res = y - z @ co
    return co[1:].T @ np.cov(z[:, 1:], rowvar=False) @ co[1:] + np.diag(
        (res * res).sum(0) / (len(X) - z.shape[1])
    )


def cvar_dual(X, w, alpha=0.1):
    # Independent dual: max z subject to capped-simplex support-function epigraph.
    # min_w max_q -q'Xw, 0<=q<=1/(alpha*T), sum q=1.
    T, n = X.shape
    # dual max lambda - cap sum u; u_i >= lambda + (X' q)_i.
    c = np.r_[np.zeros(T), -1.0, np.full(n, CAP)]
    A = np.column_stack([X.T, np.ones(n), -np.eye(n)])
    eq = np.r_[np.ones(T), np.zeros(n + 1)]
    r = linprog(
        c,
        A_ub=A,
        b_ub=np.zeros(n),
        A_eq=[eq],
        b_eq=[1],
        bounds=[(0, 1 / (alpha * T))] * T + [(None, None)] + [(0, None)] * n,
        method="highs",
    )
    if not r.success:
        raise RuntimeError(r.message)
    losses = np.sort(-X @ w)[::-1]
    k = alpha * T
    j = int(k)
    primal = (losses[:j].sum() + (k - j) * losses[j]) / k
    return float(primal + r.fun)


def inputs():
    # Cache invalidates on data, selection or loader changes (never uses cached weights).
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    paths = [
        ROOT / p
        for p in manifest["hashes"]
        if p.startswith("data/") or p.endswith("lowrisk_portfolio/data.py")
    ]
    paths += [OUT / "frozen_selection.csv", Path(__file__)]
    key = hashlib.sha256(
        (manifest["primary"] + "".join(digest(p) for p in paths)).encode()
    ).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / ("ife_implementation_audit_" + key + ".npz")
    if cache.exists():
        with np.load(cache) as archive:
            return {k: archive[k] for k in archive.files}, manifest
    print("Loading original monthly data once", flush=True)
    p = load_panel(Config())
    raw = p["total_return"]
    s = pd.read_csv(OUT / "frozen_selection.csv")
    months = sorted(s.holding_month.unique())
    xs = []
    fs = []
    ids = []
    rr = []
    ee = []
    ss = []
    allids = np.array(sorted(s.PERMNO.unique()))
    for m in months:
        t = pd.Period(m, "M")
        names = s.loc[s.holding_month.eq(m), "PERMNO"].to_numpy()
        hist = raw.loc[t - 36 : t - 1, names]
        if len(hist) != 36 or not np.isfinite(hist.to_numpy()).all():
            raise ValueError("Incomplete observed estimation window " + m)
        held = raw.loc[t, names]
        if not np.isfinite(held.to_numpy()).all():
            raise ValueError("Missing selected observed total return " + m)
        xs.append(hist.to_numpy())
        ids.append(names)
        fs.append(p.factors.loc[hist.index].to_numpy())
        realized = pd.Series(0.0, index=allids)
        realized.loc[names] = held
        rr.append(realized.to_numpy())
        ee.append(p["delisted"].loc[t, allids].to_numpy())
        ss.append(p["size"].loc[t - 1, names].to_numpy())
    np.savez_compressed(
        cache,
        X=xs,
        F=fs,
        ids=ids,
        months=months,
        allids=allids,
        R=rr,
        ended=ee,
        sizes=ss,
        factor_names=p.factors.columns.to_numpy(dtype=str),
        rf=p.factors.loc[pd.PeriodIndex(months, freq="M"), "RF"].to_numpy(),
    )
    with np.load(cache) as archive:
        return {k: archive[k] for k in archive.files}, manifest
