"""Decide whether the conditional experiments E and A should run.

Decision rule, fixed before computing: within a comparison family, a Holm-adjusted two-sided
bootstrap p below 0.05 meets a condition. Significance only before adjustment is a weak signal
and does not meet it. An A improvement must favor the portfolio with restored coskewness.
"""

import numpy as np
import pandas as pd

from common import (BOOT_BLOCK, BOOT_REPS, BOOT_SEED, CAP, REFERENCE_MODEL, RESULTS, SOURCE, covariance,
                    empty_targets, history, holm_adjust, load_inputs, module, publish, source_targets,
                    validate_weights)
from lowrisk_portfolio.metrics import drawdown
from solvers import exposure_neutral_gmvp

EXP = "check_EA_conditions"
ALPHA = 0.05
WINDOW = 36
KINDS = ["sample", "cc", "lw"]
COSKEW_ROW = 2


def indices(T):
    rng = np.random.default_rng(BOOT_SEED)
    for _ in range(BOOT_REPS):
        starts = rng.integers(0, T, size=(T + BOOT_BLOCK - 1) // BOOT_BLOCK)
        yield ((starts[:, None] + np.arange(BOOT_BLOCK)) % T).ravel()[:T]


def sharpe(excess):
    return np.sqrt(12) * excess.mean() / excess.std(ddof=1)


def cvar5(r):
    worst = np.sort(r)[: max(1, int(np.ceil(0.05 * len(r))))]
    return worst.mean()


def downside_deviation(excess):
    return np.sqrt(12 * np.mean(np.minimum(excess, 0) ** 2))


def maximum_drawdown(r):
    return float(drawdown(pd.Series(r)).min())


def bootstrap_difference(a, b, rf, stat, uses_excess):
    """Stat(a) - stat(b) on identical resampled months, with percentile interval and p."""
    x, y = (a - rf, b - rf) if uses_excess else (a, b)
    point = stat(x) - stat(y)
    draws = np.array([stat(x[i]) - stat(y[i]) for i in indices(len(a))])
    low, high = np.quantile(draws, [0.025, 0.975])
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return point, low, high, p


def holm_by(frame, *groups):
    frame["holm_p"] = np.nan
    for _, part in frame.groupby(list(groups)):
        frame.loc[part.index, "holm_p"] = holm_adjust(part.bootstrap_p.to_numpy())
    return frame


def condition_E(rf):
    grid = pd.read_csv(RESULTS / "ext01_window_cap/monthly_net_returns.csv", index_col=0)
    short = pd.read_csv(RESULTS / "ext10_short_cov_monthly_proxy/monthly_net_returns.csv", index_col=0)
    rf = rf.reindex(pd.PeriodIndex(grid.index, freq="M")).to_numpy()
    pairs = []
    for window in [36, 60]:
        for cap in ["cap2K", "cap3K", "capNone"]:
            pairs.append(("estimator_LW_vs_sample", cap, f"lw_T{window}_{cap}", f"sample_T{window}_{cap}", grid))
    for method, windows in [("sample", [60]), ("lw", [24, 60])]:
        for window in windows:
            for cap in ["cap2K", "cap3K", "capNone"]:
                pairs.append(("window_vs_36", cap, f"{method}_T{window}_{cap}", f"{method}_T36_{cap}", grid))
    for name in ["lw_T6_monthly", "lw_T12_monthly", "ewma094_monthly"]:
        pairs.append(("short_monthly_vs_lw36", "cap2K", name, "lw_T36_cap2K", short))
    rows = []
    for family, cap, a, b, frame in pairs:
        x, y = frame[a].to_numpy(), frame[b].to_numpy()
        point, low, high, p = bootstrap_difference(x, y, rf, sharpe, True)
        vol = np.sqrt(12) * (x.std(ddof=1) - y.std(ddof=1))
        rows.append(dict(family=family, cap=cap, portfolio=a, compared_to=b, delta_sharpe=point, ci_low=low,
                         ci_high=high, bootstrap_p=p, delta_annual_vol=vol))
    return holm_by(pd.DataFrame(rows), "family")


def forecast_vs_realized(inp):
    """Direct evidence of estimation error: forecast portfolio volatility against realized."""
    holdings = pd.read_csv(RESULTS / "ext01_window_cap/holdings.csv.gz")
    net = pd.read_csv(RESULTS / "ext01_window_cap/monthly_net_returns.csv", index_col=0)
    rows = []
    for kind in ["sample", "lw"]:
        for cap in ["cap2K", "cap3K", "capNone"]:
            name = f"{kind}_T{WINDOW}_{cap}"
            part = holdings[holdings.variant.eq(name)]
            forecast = []
            for m in inp.months:
                names = inp.names[m]
                w = part[part.month.eq(str(m))].set_index("permno").weight.reindex(names, fill_value=0).to_numpy()
                forecast.append(np.sqrt(12 * w @ covariance(kind, history(inp, m, WINDOW)) @ w))
            realized = np.sqrt(12) * net[name].std(ddof=1)
            rows.append(dict(variant=name, mean_forecast_vol=float(np.mean(forecast)), realized_vol=float(realized),
                             realized_over_forecast=float(realized / np.mean(forecast))))
    return pd.DataFrame(rows)


def condition_A(inp, restored, rf):
    neutral = pd.read_csv(RESULTS / "ext04_exposure_neutral/monthly_net_returns.csv", index_col=0)
    rf = rf.reindex(pd.PeriodIndex(neutral.index, freq="M")).to_numpy()
    metrics = [("sharpe", sharpe, True, 1), ("cvar5", cvar5, False, 1),
               ("downside_deviation", downside_deviation, True, -1), ("max_drawdown", maximum_drawdown, False, 1)]
    rows = []
    for kind in KINDS:
        base = neutral[f"{kind}_gmvp"].to_numpy()
        candidates = {"all_three_neutral": neutral[f"{kind}_neutral"].to_numpy(),
                      "coskew_only": restored[f"{kind}_coskew_restored"].net_return.to_numpy()}
        for restoration, values in candidates.items():
            for metric, stat, uses_excess, sign in metrics:
                point, low, high, p = bootstrap_difference(values, base, rf, stat, uses_excess)
                rows.append(dict(covariance=kind, restoration=restoration, metric=metric, difference=point,
                                 ci_low=low, ci_high=high, bootstrap_p=p, improves=bool(sign * point > 0)))
    return holm_by(pd.DataFrame(rows), "restoration")


def main():
    inp = load_inputs()
    variants = {SOURCE: dict(targets=source_targets(inp), role="baseline", method="market_cap",
                             reference_model=SOURCE)}
    diagnostics = []
    for kind in KINDS:
        print(kind, flush=True)
        base, restored = empty_targets(inp), empty_targets(inp)
        for m in inp.months:
            names = inp.names[m]
            S = covariance(kind, history(inp, m, WINDOW))
            plain = validate_weights(module("factor_cov").gmvp(S, cap=CAP), names, CAP)
            base.loc[m, names] = plain
            b = inp.source[m].to_numpy()
            A = np.vstack([np.ones(len(b)), inp.scores[m][:, COSKEW_ROW]])
            w, info = exposure_neutral_gmvp(S, A, A @ b, CAP, b)
            restored.loc[m, names] = validate_weights(w, names, CAP)
            diagnostics.append(dict(variant=f"{kind}_coskew_restored", month=str(m), **info))
        variants[f"{kind}_gmvp"] = dict(targets=base, role="baseline", method=kind, window=WINDOW, cap="cap2K",
                                        reference_model=REFERENCE_MODEL[kind])
        variants[f"{kind}_coskew_restored"] = dict(targets=restored, role="test", method=kind, window=WINDOW,
                                                   cap="cap2K", anchor=f"{kind}_gmvp")
    diagnostics = pd.DataFrame(diagnostics)
    rf = inp.panel.factors.RF
    series, _, _ = publish(
        EXP, 'Conditions for the follow-up E and A experiments',
        'Check the stated conditions using Holm-adjusted bootstrap p<0.05 within each comparison family. The coskewness-only restoration isolates the score constraint from the joint three-score restriction.',
        inp, variants, dict(max_relative_gap=float(diagnostics.relative_gap.max()), alpha=ALPHA),
        {"solver_diagnostics.csv": diagnostics},
    )
    e = condition_E(rf)
    forecast = forecast_vs_realized(inp)
    a = condition_A(inp, series, rf)
    out = RESULTS / EXP
    e.to_csv(out / "condition_E_tests.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(out / "condition_E_forecast_vs_realized.csv", index=False, encoding="utf-8-sig")
    a.to_csv(out / "condition_A_tests.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200, "display.max_rows", 100, "display.max_columns", 20)
    print(e.round(4).to_string(index=False))
    print(forecast.round(4).to_string(index=False))
    print(a.round(4).to_string(index=False))
    print("E met:", bool((e.holm_p < ALPHA).any()), e[e.holm_p < ALPHA][["portfolio", "compared_to"]].values.tolist())
    print("A met:", bool(((a.holm_p < ALPHA) & a.improves).any()),
          a[(a.holm_p < ALPHA)][["covariance", "restoration", "metric", "improves"]].values.tolist())
    print("A weak:", a[(a.bootstrap_p < ALPHA) & a.improves][["covariance", "restoration", "metric"]].values.tolist())


if __name__ == "__main__":
    main()
