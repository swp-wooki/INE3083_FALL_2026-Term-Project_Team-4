"""Validated adapters for the allocation rules used in the final report."""

import importlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "optimizers"
COVARIANCES = (
    "sample",
    "constant_corr",
    "lw_constant_corr",
    "lw_identity",
    "rmt_clipping",
    "qis",
    "ridge",
)
MODELS = (
    "equal_weight",
    "value_weight",
    *("gmvp_" + c for c in COVARIANCES),
    "gmvp_factor",
    "inverse_vol",
    "erc",
    "mdp",
    "max_sharpe_bs",
    "min_cvar",
    "mix_1n",
    "norm_gmvp",
    "tc_gmvp",
    "vol_timing",
)


def module(name):
    """Import an optimizer within this package, without changing sys.path."""
    return importlib.import_module(".optimizers." + name, package=__package__)


def validate_weights(weights, names, cap):
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(names),) or not np.isfinite(w).all():
        raise ValueError("Invalid optimizer output shape or nonfinite weights")
    if abs(w.sum() - 1) > 1e-6 or w.min() < -1e-7 or w.max() > cap + 1e-6:
        raise ValueError("Optimizer violated long-only/budget/cap constraint")
    w = np.maximum(w, 0)
    return pd.Series(w / w.sum(), index=names)


def allocate(name, history, sizes, factors, previous, cap, cost):
    """history/factors end at formation t; previous is drifted wealth before rebalancing."""
    ids = history.columns
    n = len(ids)
    X = history.to_numpy()
    if name == "equal_weight":
        w = np.full(n, 1 / n)
    elif name == "value_weight":
        w = module("factor_cov")._proj((sizes / sizes.sum()).to_numpy(), cap)
    elif name.startswith("gmvp_"):
        kind = name.removeprefix("gmvp_")
        if kind == "factor":
            # FF5 regressions are on excess stock returns, with identical monthly dates.
            S = (
                module("factor_cov")
                .factor_cov(history.sub(factors.RF, axis=0), factors.drop(columns="RF"))
                .to_numpy()
            )
        else:
            fn = module("cov_" + kind).estimate
            S = fn(X, legacy=False) if kind == "lw_constant_corr" else fn(X)
        if not np.isfinite(S).all() or np.linalg.eigvalsh(S).min() < -1e-8:
            raise ValueError("Nonfinite or indefinite covariance")
        # Same supplied convergent solver for every covariance estimator.
        w = module("factor_cov").gmvp(S, cap=cap)
    else:
        S = np.cov(X, rowvar=False)
        # With N > T-1 the sample covariance is singular. The supplied certified
        # MDP/ERC solvers require positive definiteness. Use a disclosed numerical
        # eigenvalue floor only for these solvers, not statistical shrinkage.
        if name in {"mdp", "erc"}:
            S = positive_definite_floor(S)
        fun = getattr(module(name), name)
        if name == "max_sharpe_bs":
            shrink = module("cov_lw_identity").estimate(X)
            w = fun(history, cov=shrink, cap=cap, rf=float(factors.RF.iloc[-1]))
        elif name == "min_cvar":
            # The supplied function silently falls back if its LP fails. Surface that failure.
            mod = module(name)
            original = mod.linprog

            def checked(*args, **kwargs):
                result = original(*args, **kwargs)
                if not result.success:
                    raise RuntimeError("CVaR optimization failed: " + result.message)
                return result

            with temporary_attribute(mod, "linprog", checked):
                w = fun(history, cap=cap, alpha=0.10)
        elif name == "tc_gmvp":
            w = fun(
                S,
                w_prev=previous.reindex(ids, fill_value=0).to_numpy(),
                cost=cost,
                gamma=5.0,
                cap=cap,
            )
        elif name == "mix_1n":
            w = fun(cov=S, a=0.5, cap=cap)
        elif name == "norm_gmvp":
            w = fun(cov=S, min_eff=0.5 * n, cap=cap)
        else:
            w = fun(cov=S, cap=cap)
    return validate_weights(w, ids, cap)


def positive_definite_floor(cov):
    eigenvalues = np.linalg.eigvalsh(cov)
    floor = max(float(eigenvalues[-1]) * 1e-10, 1e-14)
    ridge = max(0.0, floor - float(eigenvalues[0]))
    return cov + ridge * np.eye(len(cov)) if ridge else cov


@contextmanager
def temporary_attribute(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)
