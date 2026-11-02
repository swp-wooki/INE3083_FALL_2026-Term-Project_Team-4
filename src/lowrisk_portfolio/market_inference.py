"""Market inference from saved baseline returns; no selection or allocation refit.

Circular-block percentile intervals condition on the chosen strategy. The fraction
of unrestricted bootstrap estimates at or below zero is not a null-imposed p-value.
FF5 intercept inference uses Bartlett HAC(4), without a degrees-of-freedom factor,
matching the article's existing HAC convention. Annual alpha is 12 times intercept.
"""

from .paths import OUTPUT, ROOT

P = OUTPUT
OUT = OUTPUT
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import read_factors

PERIODS = {
    "full": ("2003-02", "2020-12"),
    "2003_2011": ("2003-02", "2011-12"),
    "2012_2020": ("2012-01", "2020-12"),
}
MARKETS = ["FF_market", "CRSP_market", "SP500"]
SEED, REPLICATIONS, HAC_LAGS = 20261002, 2000, 4


def alpha_hac(y, factors):
    """OLS/HAC with explicit contractions, plus an independent scalar SE check."""
    x = np.column_stack([np.ones(len(y)), factors])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - np.einsum("ij,j->i", x, beta)
    z = x * residual[:, None]
    bread = np.linalg.inv(np.einsum("ni,nj->ij", x, x))
    meat = np.einsum("ni,nj->ij", z, z)
    for lag in range(1, HAC_LAGS + 1):
        cross = np.einsum("ni,nj->ij", z[lag:], z[:-lag])
        meat += (1 - lag / (HAC_LAGS + 1)) * (cross + cross.T)
    variance = np.einsum("i,ij,j->", bread[0], meat, bread[0])
    influence = np.einsum("ni,i->n", z, bread[0])
    scalar_variance = sum(float(v) ** 2 for v in influence)
    for lag in range(1, HAC_LAGS + 1):
        scalar_variance += (
            2
            * (1 - lag / (HAC_LAGS + 1))
            * sum(float(influence[t]) * float(influence[t - lag]) for t in range(lag, len(y)))
        )
    np.testing.assert_allclose(variance, scalar_variance, rtol=1e-11)
    assert variance > 0 and np.isfinite(beta).all()
    t = float(beta[0] / np.sqrt(variance))
    return float(12 * beta[0]), t, float(2 * norm.sf(abs(t)))


def run():
    inputs = [
        P / "monthly_returns.csv",
        P / "benchmark_returns.csv",
        ROOT / "data/raw/factor_returns_monthly.csv",
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    returns = pd.read_csv(inputs[0])
    strategy = (
        returns.loc[
            returns.scenario.eq("total_observed_drift10") & returns.model.eq("source_weight")
        ]
        .set_index("month")
        .net_return.sort_index()
    )
    benchmark = pd.read_csv(inputs[1], index_col="month").sort_index()
    factors = read_factors(inputs[2])
    factors.index = factors.index.astype(str)
    assert strategy.index.is_unique and len(strategy) == 215
    assert strategy.index.equals(benchmark.index)
    np.testing.assert_allclose(benchmark.RF, factors.loc[strategy.index, "RF"], atol=1e-15)
    np.testing.assert_allclose(
        benchmark.FF_market,
        factors.loc[strategy.index, "Mkt-RF"] + benchmark.RF,
        atol=1e-15,
    )
    saved = (
        pd.read_csv(P / "model_metrics.csv")
        .query('scenario == "total_observed_drift10" and model == "source_weight"')
        .set_index("period")
    )
    rows = []
    for period, (start, end) in PERIODS.items():
        r = strategy.loc[start:end]
        data = pd.concat([r.rename("strategy"), benchmark.loc[r.index, MARKETS]], axis=1)
        excess = data.to_numpy() - benchmark.loc[r.index, "RF"].to_numpy()[:, None]
        assert np.isfinite(excess).all()
        n = len(r)
        observed = np.sqrt(12) * excess.mean(axis=0) / excess.std(axis=0, ddof=1)
        np.testing.assert_allclose(observed[0], saved.loc[period, "sharpe"], atol=1e-12)
        ff3 = alpha_hac(excess[:, 0], factors.loc[r.index, ["Mkt-RF", "SMB", "HML"]].to_numpy())
        np.testing.assert_allclose(
            ff3,
            saved.loc[period, ["ff3_alpha_annual", "ff3_alpha_hac_t", "ff3_alpha_hac_p"]].to_numpy(
                float
            ),
            atol=1e-11,
        )
        alpha, t, p = alpha_hac(
            excess[:, 0],
            factors.loc[r.index, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].to_numpy(),
        )
        for block in [3, 6, 12]:
            rng = np.random.default_rng(SEED)
            draws = np.empty((REPLICATIONS, len(MARKETS)))
            for j in range(REPLICATIONS):
                starts = rng.integers(0, n, size=(n + block - 1) // block)
                index = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
                sample = excess[index]
                sr = np.sqrt(12) * sample.mean(axis=0) / sample.std(axis=0, ddof=1)
                draws[j] = sr[0] - sr[1:]
            lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
            for j, market in enumerate(MARKETS):
                rows.append(
                    dict(
                        period=period,
                        start=start,
                        end=end,
                        months=n,
                        market=market,
                        strategy_sharpe=observed[0],
                        market_sharpe=observed[j + 1],
                        delta_sharpe=observed[0] - observed[j + 1],
                        ci95_lower=lower[j],
                        ci95_upper=upper[j],
                        bootstrap_nonpositive_fraction=float(np.mean(draws[:, j] <= 0)),
                        ff5_alpha_annual=alpha,
                        ff5_alpha_hac_t=t,
                        ff5_alpha_hac_p=p,
                        block_months=block,
                        replications=REPLICATIONS,
                        seed=SEED,
                        hac_lags=HAC_LAGS,
                        round_trip_cost_bps=20,
                        returns_sha256=hashes[inputs[0].name],
                        benchmarks_sha256=hashes[inputs[1].name],
                        factors_sha256=hashes[inputs[2].name],
                    )
                )
    result = pd.DataFrame(rows)
    result.to_csv(P / "market_inference.csv", index=False)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    print(
        result.loc[
            result.market.eq("FF_market"),
            [
                "period",
                "block_months",
                "delta_sharpe",
                "ci95_lower",
                "ci95_upper",
                "ff5_alpha_annual",
                "ff5_alpha_hac_t",
                "ff5_alpha_hac_p",
            ],
        ]
        .round(6)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()
