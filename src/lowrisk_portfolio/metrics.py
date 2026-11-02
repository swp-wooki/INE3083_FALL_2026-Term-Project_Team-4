"""Decimal monthly returns; annualization uses 12. No percent/decimal ambiguity."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def hac_regression(y, x=None, lags=4):
    """OLS with Bartlett/Newey-West covariance, asymptotic two-sided normal p-values."""
    y = np.asarray(y, float)
    x = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    z = x * (y - x @ beta)[:, None]
    meat = z.T @ z
    for lag in range(1, min(lags, len(y) - 1) + 1):
        g = z[lag:].T @ z[:-lag]
        meat += (1 - lag / (lags + 1)) * (g + g.T)
    bread = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(bread @ meat @ bread), 0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return beta, t, 2 * norm.sf(np.abs(t))


def drawdown(r):
    wealth = (1 + r).cumprod()
    peak = wealth.cummax().clip(lower=1)
    return wealth / peak - 1


def performance(r, rf, market=None, factors=None, lags=4):
    r = pd.Series(r).astype(float)
    rf = rf.reindex(r.index)
    if r.isna().any() or rf.isna().any() or len(r) < 2 or r.le(-1).any():
        raise ValueError("Metrics require >=2 complete, solvent, aligned months")
    excess = r - rf
    vol = r.std(ddof=1) * np.sqrt(12)
    sd = excess.std(ddof=1)
    wealth = float((1 + r).prod())
    out = {
        "months": len(r),
        "cumulative_return": wealth - 1,
        "cagr": wealth ** (12 / len(r)) - 1,
        "annual_mean_return": r.mean() * 12,
        "report_annual_return": (1 + r.mean()) ** 12 - 1,
        "annual_volatility": vol,
        "sharpe": float(excess.mean() / sd * np.sqrt(12)) if sd else np.nan,
        "max_drawdown": float(drawdown(r).min()),
        "positive_month_fraction": float(r.gt(0).mean()),
    }
    if market is not None:
        b = market.reindex(r.index)
        if b.isna().any():
            raise ValueError("Missing benchmark month")
        active = r - b
        _, t, p = hac_regression(active, lags=lags)
        out.update(
            active_return_monthly=active.mean(),
            active_return_hac_t=t[0],
            active_return_hac_p=p[0],
            tracking_error=active.std(ddof=1) * np.sqrt(12),
        )
    if factors is not None:
        f = factors.reindex(r.index)
        if f.isna().any().any():
            raise ValueError("Missing factor month")
        beta, t, p = hac_regression(excess, f[["Mkt-RF", "SMB", "HML"]].to_numpy(), lags)
        out.update(
            ff3_alpha_annual=12 * beta[0],
            ff3_alpha_hac_t=t[0],
            ff3_alpha_hac_p=p[0],
            market_beta=beta[1],
        )
    return out
