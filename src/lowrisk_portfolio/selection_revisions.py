"""Reconstruct the matched selection experiments and temporal factor controls.

The historical selector uses adjusted-price returns, its classified terminal-return
policy, and nominal turnover. Do not substitute allocation-engine return series
for these reselected portfolios. Appendix A explicitly identifies the two panels.
"""

import json
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .data import read_factors
from .metrics import hac_regression
from .paths import DATA, OUTPUT, ROOT

SEED = 20261109
DRAWS = 2000
BLOCK = 6
PERIODS = {
    "Full": ("2003-02", "2020-12"),
    "2003--11": ("2003-02", "2011-12"),
    "2012--20": ("2012-01", "2020-12"),
}
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
SIGNALS = {
    "B": "low_beta_fp",
    "V": "low_vol_12m",
    "C": "coskew",
    "M": "mom_12_1",
    "L": "high_52w_low",
    "P": "rmw_beta",
    "I": "net_issuance",
}
ADDITIONS = {
    "BVC": "Defensive, three signals",
    "BVCM": "Add 12--1 momentum",
    "BVCL": "Add distance from 52-week low",
    "BVCP": "Add profitability beta",
    "BVCI": "Add net issuance",
    "RO": "Return-oriented, 50 stocks",
    "80/20": "80/20 blend",
    "50/50": "50/50 blend",
}
GROUPS = ["B", "V", "C", "BV", "BC", "VC", "BVC"]


def reconstruct(out):
    """Replay the supplied selection algorithms; output one row per holding month."""
    sys.path.insert(0, str(ROOT / "selection"))
    os.environ["SELECTION_OUTPUT_DIR"] = str(out.parent / "selection")
    os.environ["FR_PROFILE"] = "cap2"
    import factor_research_v2 as fr
    import selection_eval as se

    # Fresh source inputs: a stale, unversioned MarketData pickle must never win.
    market = se.load_all(delist_policy="classified")
    cfg = se.Config(
        vol_cut=0,
        size_cut=0.20,
        size_basis="nyse",
        weighting="vw",
        exclude_delist_month=True,
        vw_cap_mult=2,
    )
    fr._RANKS.clear()
    ranks = {k: fr.ranks_of(name, market, cfg) for k, name in SIGNALS.items()}

    def weights(key, k=30, buffer=1.5, industry=0.30):
        score = sum(ranks[letter] for letter in key) / len(key)
        return fr.topk_weights(score, market, k, cfg=cfg, buffer=buffer, ind_cap=industry)

    returns = {}
    betas = {}
    months = cfg.form_months() + 1
    for key in (*GROUPS, "BVCM", "BVCL", "BVCP", "BVCI", "RO"):
        w = weights(key) if key != "RO" else weights("LPI", 50, industry=None)
        returns[key] = se.portfolio_returns(w, market, cfg).net
        if key in ("BVC", "RO"):
            name = "defensive" if key == "BVC" else "return_oriented"
            expected = (
                pd.read_csv(DATA / f"selections/{name}_selection.csv")
                .set_index(["formation_month", "PERMNO"])
                .weight.sort_index()
            )
            actual = w.rename_axis(index="formation_month", columns="PERMNO").copy()
            actual.index = actual.index.astype(str)
            actual = actual.where(actual.gt(0)).stack().sort_index()
            if not expected.index.equals(actual.index):
                raise ValueError(
                    f"Reconstructed {name} constituents differ from the supplied selection"
                )
            np.testing.assert_allclose(actual.round(6), expected, rtol=0, atol=1e-12)
        if key in ("BVC", "BV"):
            t = pd.Period("2020-01", "M")
            held = w.loc[t][w.loc[t] > 0]
            hist = market.ret.loc[t - 35 : t, held.index]
            f = market.ff.loc[hist.index]
            x = np.column_stack([np.ones(len(f)), f[FACTORS]])
            slopes = []
            counts = []
            for name in held.index:
                y = (hist[name] - f.RF).to_numpy()
                valid = np.isfinite(y)
                if valid.sum() < 7 or np.linalg.matrix_rank(x[valid]) != x.shape[1]:
                    raise ValueError(f"Insufficient pre-crisis observations for {name}")
                slopes.append(np.linalg.lstsq(x[valid], y[valid], rcond=None)[0][1])
                counts.append(int(valid.sum()))
            betas[key] = {
                "market_beta": float(held @ slopes),
                "min_observations": min(counts),
                "max_observations": max(counts),
            }
    # Both sleeves are rebalanced monthly to the stated allocation; each leg's net return already includes its trading costs.
    for label, weight in [("80/20", 0.8), ("50/50", 0.5)]:
        returns[label] = weight * returns["BVC"] + (1 - weight) * returns["RO"]
    for label, k, b, i in [(f"K{k}", k, 1.5, 0.30) for k in [20, 40, 50, 100]] + [
        ("no_buffer", 30, None, 0.30),
        ("no_industry", 30, 1.5, None),
        ("neither", 30, None, None),
    ]:
        for key in ("BV", "BVC"):
            returns[f"{key}_{label}"] = se.portfolio_returns(weights(key, k, b, i), market, cfg).net
    # Gross, capitalization-weighted 30/30 factor portfolios; no stock cap on the long and short legs.
    for key in ("C", "V"):
        r = ranks[key]
        cap = market.x["mcap"].reindex_like(r)
        hi = se.weigh(r.ge(0.70), cap)
        lo = se.weigh(r.le(0.30), cap)
        returns["spread_" + key] = (
            se.portfolio_returns(hi, market, cfg).gross
            - se.portfolio_returns(lo, market, cfg).gross
        )
    # Cross-sectional slopes of next-month observed adjusted-price returns, simultaneous B,V,C ranks and an intercept.
    slopes = []
    for t in cfg.form_months():
        panel = (
            pd.DataFrame({k: ranks[k].loc[t] for k in "BVC"})
            .assign(forward=market.ret.loc[t + 1])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        x = np.column_stack([np.ones(len(panel)), panel[list("BVC")]])
        slopes.append(np.linalg.lstsq(x, panel.forward, rcond=None)[0][1:])
    frame = pd.DataFrame(returns).join(
        pd.DataFrame(slopes, index=months, columns=["slope_" + k for k in "BVC"])
    )
    frame = frame.join(market.ff.loc[months, [*FACTORS, "RF", "MKT"]])
    frame.index.name = "month"
    if not frame.index.equals(months) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(
            "Missing or misaligned selection return, cross-sectional slope, or factor observation"
        )
    frame.to_csv(out / "selection_monthly.csv")
    (out / "precrisis_beta.json").write_text(json.dumps(betas, indent=2) + "\n")
    return frame


def _sr(excess, axis=0):
    return np.sqrt(12) * np.mean(excess, axis=axis) / np.std(excess, axis=axis, ddof=1)


def _blocks(n, rng, draws=DRAWS, block=BLOCK):
    starts = rng.integers(n, size=(draws, (n + block - 1) // block))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def bootstrap_precision(frame, draws=100_000, seed=91007, chunk=500):
    """Check two fourth-signal intervals without materializing a draws-by-month panel."""
    if draws < 2 or chunk < 1:
        raise ValueError("At least two draws and a positive chunk size are required")
    rng = np.random.default_rng(seed)
    models = ("BVCL", "BVCP")
    xs = {k: (frame[k] - frame.RF).to_numpy() for k in (*models, "BVC")}
    differences = np.empty((draws, len(models)))
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        idx = _blocks(len(frame), rng, draws=size)
        baseline = _sr(xs["BVC"][idx], axis=1)
        for j, k in enumerate(models):
            differences[start : start + size, j] = _sr(xs[k][idx], axis=1) - baseline
    intervals = np.quantile(differences, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        [
            dict(
                model=k,
                seed=seed,
                draws=draws,
                block_months=BLOCK,
                ci_lo=intervals[0, j],
                ci_hi=intervals[1, j],
            )
            for j, k in enumerate(models)
        ]
    )


def _growth(r):
    return np.prod(1 + r) - 1


def _metrics(r, rf, mkt):
    wealth = np.r_[1.0, np.cumprod(1 + r.to_numpy())]
    excess = r - rf
    return dict(
        cagr=100 * np.expm1(np.log1p(r).mean() * 12),
        vol=100 * r.std() * np.sqrt(12),
        sharpe=_sr(excess),
        mdd=100 * np.min(wealth / np.maximum.accumulate(wealth) - 1),
        beta=hac_regression(excess, mkt)[0][1],
    )


def _interaction(y, f, post, lag):
    """Fully interacted FF5 model: Bartlett HAC covariance for a joint Wald test."""
    x = np.column_stack([np.ones(len(y)), f, post, f * post[:, None]])
    co = np.linalg.lstsq(x, y, rcond=None)[0]
    score = x * (y - x @ co)[:, None]
    meat = score.T @ score
    for j in range(1, lag + 1):
        cross = score[j:].T @ score[:-j]
        meat += (1 - j / (lag + 1)) * (cross + cross.T)
    inv = np.linalg.pinv(x.T @ x)
    cov = inv @ meat @ inv
    delta = co[7:]
    wald = delta @ np.linalg.solve(cov[7:, 7:], delta)
    return {
        "lags": lag,
        "alpha_change": 1200 * co[6],
        "alpha_change_t": co[6] / np.sqrt(cov[6, 6]),
        "market_change": co[7],
        "cma_change": co[-1],
        "wald_p": chi2.sf(wald, 5),
    }


def temporal(out, frame):
    raw = pd.read_csv(out.parent / "monthly_returns.csv")
    raw = (
        raw.loc[(raw.model == "source_weight") & (raw.scenario == "total_observed_drift10")]
        .set_index("month")
        .net_return
    )
    raw.index = pd.PeriodIndex(raw.index, freq="M")
    raw = raw.sort_index()
    if not raw.index.equals(frame.index):
        raise ValueError(
            "Original allocation sample differs from the reconstructed selection months"
        )
    original = read_factors(DATA / "raw/factor_returns_monthly.csv").loc[raw.index].copy()
    bab = pd.read_csv(ROOT / "extensions/external_data/aqr_bab_usa.csv")
    bab.index = pd.PeriodIndex(bab.month, freq="M")
    if not bab.index.is_unique:
        raise ValueError("Duplicate BAB factor months")
    original["BAB"] = bab.BAB_USA.reindex(raw.index)
    frame = frame.assign(BAB=original.BAB)
    if original.isna().any().any():
        raise ValueError("Missing original FF5/BAB observations")
    y = (raw - original.RF).to_numpy()
    f = original[FACTORS].to_numpy()
    post = (raw.index >= pd.Period("2012-01", "M")).astype(float)
    interaction = pd.DataFrame([_interaction(y, f, post, lag) for lag in (4, 6, 12)])
    interaction.to_csv(out / "temporal_interactions.csv", index=False)
    rolling = []
    for window in (36, 60):
        for end in range(window, len(y) + 1):
            co = np.linalg.lstsq(
                np.column_stack([np.ones(window), f[end - window : end]]),
                y[end - window : end],
                rcond=None,
            )[0]
            rolling.append(
                dict(
                    window=window,
                    ending=str(raw.index[end - 1]),
                    alpha=1200 * co[0],
                    market_beta=co[1],
                )
            )
    rolling = pd.DataFrame(rolling)
    rolling.to_csv(out / "temporal_rolling.csv", index=False)
    # Preserve each pre/post-crisis segment’s length and do not bridge the omitted crisis.
    benchmarks = pd.read_csv(out.parent / "benchmark_returns.csv", index_col="month")
    benchmarks.index = pd.PeriodIndex(benchmarks.index, freq="M")
    benchmarks = benchmarks.loc[raw.index]
    crisis = (raw.index >= pd.Period("2007-07", "M")) & (raw.index <= pd.Period("2009-03", "M"))
    segments = [
        np.flatnonzero(raw.index < pd.Period("2007-07", "M")),
        np.flatnonzero(raw.index > pd.Period("2009-03", "M")),
    ]
    generator = np.random.default_rng(SEED)
    indices = np.concatenate([sub[_blocks(len(sub), generator)] for sub in segments], axis=1)
    crisis_results = []
    captures = []
    for market in ["CRSP_market", "FF_market"]:
        bench = benchmarks[market]
        excess = (bench - original.RF).to_numpy()
        difference = np.log1p(raw) - np.log1p(bench)
        lo, hi = np.quantile(_sr(y[indices], axis=1) - _sr(excess[indices], axis=1), [0.025, 0.975])
        crisis_results.append(
            dict(
                market=market,
                total_log_points=100 * difference.sum(),
                crisis_log_points=100 * difference[crisis].sum(),
                other_months=int((~crisis).sum()),
                other_sharpe_difference=_sr(y[~crisis]) - _sr(excess[~crisis]),
                ci_lo=lo,
                ci_hi=hi,
            )
        )
    for period, start, end in [
        ("2007_decline", "2007-11", "2009-02"),
        ("2009_rebound", "2009-03", "2009-05"),
        ("2020_decline", "2020-02", "2020-03"),
        ("2020_rebound", "2020-04", "2020-06"),
    ]:
        for strategy, series, market in [
            ("original", raw, benchmarks.FF_market),
            ("BV", frame.BV, frame.MKT),
            ("BVC", frame.BVC, frame.MKT),
        ]:
            r = series.loc[start:end]
            b = market.loc[start:end]
            captures.append(
                dict(
                    period=period,
                    strategy=strategy,
                    months=len(r),
                    capture_pct=100 * _growth(r) / _growth(b),
                )
            )
    pd.DataFrame(crisis_results).to_csv(out / "crisis_contribution.csv", index=False)
    pd.DataFrame(captures).to_csv(out / "capture_windows.csv", index=False)
    regressions = []
    means = []
    for period, (a, b) in PERIODS.items():
        s = frame.loc[a:b]
        o = original.loc[a:b]
        orig_ex = raw.loc[a:b] - o.RF
        for name, ex, fs, cols in [
            ("original_ff5", orig_ex, o, FACTORS),
            ("original_ff5_bab", orig_ex, o, [*FACTORS, "BAB"]),
            ("rerun_ff5_bab", s.BVC - s.RF, s, [*FACTORS, "BAB"]),
            ("rerun_ff5_bab_lowvol", s.BVC - s.RF, s, [*FACTORS, "BAB", "spread_V"]),
            ("coskew_ff5_bab", s.spread_C, s, [*FACTORS, "BAB"]),
        ]:
            co, t, _ = hac_regression(ex, fs[cols])
            row = dict(model=name, period=period, alpha=1200 * co[0], alpha_t=t[0])
            row.update({k: v for k, v in zip(cols, co[1:])})
            row.update({k + "_t": v for k, v in zip(cols, t[1:])})
            regressions.append(row)
        means.append(dict(period=period, bab=1200 * s.BAB.mean(), coskew=1200 * s.spread_C.mean()))
    regressions = pd.DataFrame(regressions)
    regressions.to_csv(out / "temporal_regressions.csv", index=False)
    pd.DataFrame(means).to_csv(out / "temporal_factor_means.csv", index=False)


def summarize(out):
    frame = pd.read_csv(out / "selection_monthly.csv", index_col="month")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    period_metrics = []
    paired = []
    fmb = []
    draws = {}
    removals = []
    contrasts = []
    rng = np.random.default_rng(SEED)
    variants = [*GROUPS, *(s for s in ADDITIONS if s != "BVC")]
    for period, (a, b) in PERIODS.items():
        s = frame.loc[a:b]
        ix = _blocks(len(s), rng)
        draw = {}
        for model in variants:
            r = s[model]
            e = (r - s.RF).to_numpy()
            draw[model] = _sr(e[ix], axis=1)
            result = _metrics(r, s.RF, s["Mkt-RF"])
            period_metrics.append(dict(period=period, model=model, **result))
        for k, removed in [("B", "VC"), ("V", "BC"), ("C", "BV")]:
            change = draw["BVC"] - draw[removed]
            lo, hi = np.quantile(change, [0.025, 0.975])
            removals.append(
                dict(
                    period=period,
                    removed=k,
                    delta=_sr(s.BVC - s.RF) - _sr(s[removed] - s.RF),
                    lo=lo,
                    hi=hi,
                )
            )
        for (k1, m1), (k2, m2) in combinations([("B", "VC"), ("V", "BC"), ("C", "BV")], 2):
            lo, hi = np.quantile(draw[m2] - draw[m1], [0.025, 0.975])
            contrasts.append(dict(period=period, removed_1=k1, removed_2=k2, lo=lo, hi=hi))
        co, t, _ = hac_regression(s.BVC - s.BV)
        span, span_t, _ = hac_regression(s.BVC - s.RF, np.column_stack([s.BV - s.RF, s[FACTORS]]))
        delta = draw["BVC"] - draw["BV"]
        lo, hi = np.quantile(delta, [0.025, 0.975])
        draws[period] = delta
        paired.append(
            dict(
                period=period,
                mean_diff=1200 * co[0],
                hac_t=t[0],
                delta_sr=_sr(s.BVC - s.RF) - _sr(s.BV - s.RF),
                ci_lo=lo,
                ci_hi=hi,
                alpha=1200 * span[0],
                alpha_t=span_t[0],
            )
        )
        for k in "BVC":
            co, t, _ = hac_regression(s["slope_" + k])
            fmb.append(dict(period=period, signal=k, slope_monthly_pct=100 * co[0], hac_t=t[0]))
        if period == "Full":
            tab9 = []
            for k in ADDITIONS:
                lo, hi = np.quantile(draw[k] - draw["BVC"], [0.025, 0.975])
                tab9.append(
                    dict(
                        model=k,
                        **_metrics(s[k], s.RF, s["Mkt-RF"]),
                        delta_sr=_sr(s[k] - s.RF) - _sr(s.BVC - s.RF),
                        ci_lo=lo,
                        ci_hi=hi,
                    )
                )
    for name, data in [
        ("signal_metrics", period_metrics),
        ("signal_increment", paired),
        ("signal_removals", removals),
        ("removal_contrasts", contrasts),
        ("joint_fama_macbeth", fmb),
        ("fourth_signals_blends", tab9),
    ]:
        pd.DataFrame(data).to_csv(out / (name + ".csv"), index=False)
    d = frame.BVC - frame.BV
    co, t, _ = hac_regression(d, (d.index >= pd.Period("2012-01", "M")).astype(float))
    changes = dict(
        mean_diff=1200 * co[1],
        t=t[1],
        sr_change=paired[2]["delta_sr"] - paired[1]["delta_sr"],
        ci=list(np.quantile(draws["2012--20"] - draws["2003--11"], [0.025, 0.975])),
    )
    (out / "signal_period_change.json").write_text(json.dumps(changes, indent=2) + "\n")
    downside = []
    m = frame.MKT
    crisis = m.loc["2007-07":"2009-03"]
    worst = crisis.nsmallest(7).index
    for key in ["BV", "BVC"]:
        r = frame[key]
        neg = frame["Mkt-RF"] < 0
        downside.append(
            dict(
                model=key,
                downside_beta=hac_regression((r - frame.RF)[neg], frame["Mkt-RF"][neg])[0][1],
                bottom_decile_n=len(m[m <= m.quantile(0.10)]),
                bottom_decile_mean=100 * r[m <= m.quantile(0.10)].mean(),
                worst_five_mean=100 * r.loc[m.nsmallest(5).index].mean(),
                crisis_worst_seven=100 * _growth(r.loc[worst]),
                crisis_other_fourteen=100 * _growth(r.loc[crisis.index.difference(worst)]),
            )
        )
    pd.DataFrame(downside).to_csv(out / "downside_risk.csv", index=False)
    sens = []
    for label in ["K20", "K30", "K40", "K50", "K100", "no_buffer", "no_industry", "neither"]:
        v = {k: frame[k if label == "K30" else k + "_" + label] for k in ["BV", "BVC"]}
        z = {k: _metrics(r, frame.RF, frame["Mkt-RF"]) for k, r in v.items()}
        sens.append(
            dict(
                setting=label,
                delta_sr=z["BVC"]["sharpe"] - z["BV"]["sharpe"],
                bv_mdd=z["BV"]["mdd"],
                bvc_mdd=z["BVC"]["mdd"],
            )
        )
    pd.DataFrame(sens).to_csv(out / "selection_sensitivity.csv", index=False)
    bootstrap_precision(frame).to_csv(out / "bootstrap_precision.csv", index=False)
    temporal(out, frame)
    (out / "selection_methods.json").write_text(
        json.dumps(
            dict(
                bootstrap=dict(
                    draws=DRAWS,
                    block_months=BLOCK,
                    seed=SEED,
                    method="paired circular blocks with contiguous period-specific resampling",
                ),
                hac="Bartlett, four lags, asymptotic covariance without degrees-of-freedom adjustment",
                long_short="Gross capitalization-weighted top/bottom 30 percent within eligible ranks; no stock cap",
                blend="Fixed monthly mixture of net strategy sleeve returns; sleeve turnover charged separately",
                downside="OLS excess strategy return on FF excess market in negative excess-market months; tail months ranked on FF total market",
                cross_section="Equal-weighted cross-sectional OLS on joint fractional ranks; next observed adjusted-price return",
                source_comparison="Supplied defensive and return-oriented constituents and rounded weights checked during reconstruction",
            ),
            indent=2,
        )
        + "\n"
    )
    write_rows(out)


def write_rows(out):
    m = pd.read_csv(out / "signal_metrics.csv").set_index(["period", "model"])
    a = []
    b = []
    c = []
    for k in GROUPS:
        vals = [f"{m.loc[p, k].sharpe:.2f}" for p in PERIODS]
        z = m.loc["Full", k]
        dd = f"{z.mdd:.1f}"
        if k == "BVC":
            vals[0] = r"\textbf{" + vals[0] + "}"
        if k == "C":
            dd = r"\underline{" + dd + "}"
        a.append(
            " & ".join(["$" + "+".join(k) + "$", *vals, f"{z.vol:.1f}", f"${dd}$", f"{z.beta:.2f}"])
            + r"\\"
        )
    for v in pd.read_csv(out / "signal_increment.csv").itertuples():
        delta = f"{v.delta_sr:.3f}"
        ci = f"[{v.ci_lo:.3f},{v.ci_hi:.3f}]"
        if v.period == "Full":
            delta = r"\textbf{" + delta + "}"
        if v.period == "2012--20":
            ci = r"\underline{" + ci + "}"
        b.append(
            f"{v.period} & "
            + r"\multicolumn{2}{c}{"
            + f"{v.mean_diff:.2f} ({v.hac_t:.2f})"
            + "} & "
            + delta
            + " & "
            + r"\multicolumn{2}{c}{$"
            + ci
            + r"$} & "
            + f"{v.alpha:.2f} ({v.alpha_t:.2f})"
            + r"\\"
        )
    for v in pd.read_csv(out / "fourth_signals_blends.csv").itertuples():
        ca = f"{v.cagr:.2f}"
        dd = f"${v.mdd:.2f}$"
        ds = f"${v.delta_sr:+.3f}$"
        ci = f"$[{v.ci_lo:.3f},{v.ci_hi:.3f}]$"
        if v.model == "BVC":
            ds = ci = "---"
        if v.model == "80/20":
            ca = r"\textbf{" + ca + "}"
            dd = r"\underline{" + dd + "}"
        c.append(
            " & ".join([ADDITIONS[v.model], ca, f"{v.vol:.2f}", f"{v.sharpe:.3f}", dd, ds, ci])
            + r"\\"
            + (r"\midrule" if v.model == "BVCI" else "")
        )
    for name, rows in [("archive_a", a), ("archive_b", b), ("selectionrisk", c)]:
        (out / (name + ".tex")).write_text("\n".join(rows) + "\n")


def main():
    out = OUTPUT / "revisions"
    out.mkdir(parents=True, exist_ok=True)
    reconstruct(out)
    summarize(out)


def _paragraph(text, start, body):
    """Replace one reviewed paragraph; fail on an ambiguous or missing anchor."""
    pattern = "^" + re.escape(start) + r"[^\n]*$"
    result, count = re.subn(pattern, lambda _: body, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one manuscript paragraph beginning {start!r}; found {count}")
    return result


def render_prose(text, revisions):
    """Bind the abstract, signal comparisons, downside diagnostics, and Appendix A to outputs."""

    def table(name, index):
        return pd.read_csv(revisions / (name + ".csv")).set_index(index)

    inc = table("signal_increment", "period")
    full = inc.loc["Full"]
    change = json.loads((revisions / "signal_period_change.json").read_text())
    downside = table("downside_risk", "model")
    without = downside.loc["BV"]
    with_c = downside.loc["BVC"]
    removal = table("signal_removals", ["period", "removed"]).loc["Full"]
    fmb = table("joint_fama_macbeth", ["period", "signal"])
    sensitivity = table("selection_sensitivity", "setting")
    adds = table("fourth_signals_blends", "model")
    a = adds.loc["80/20"]
    base = adds.loc["BVC"]
    tmp = table("temporal_interactions", "lags")
    reg = table("temporal_regressions", ["model", "period"])
    roll = table("temporal_rolling", ["window", "ending"])
    means = table("temporal_factor_means", "period")
    lower, upper = change["ci"]
    # The sentence also occurs in the abstract: narrow the edit to the interval and betas.
    abstract = re.compile(
        r"(Adding negative coskewness to low beta and low volatility raises the full-period Sharpe ratio by )[-\d.]+ \(95\\% confidence interval [-\d.]+ to [-\d.]+\), but also raises downside beta from [-\d.]+ to \d+\.\d+\."
    )
    text, n = abstract.subn(
        lambda m: (
            m.group(1)
            + rf"{full.delta_sr:.3f} (95\% confidence interval {full.ci_lo:.3f} to {full.ci_hi:.3f}), but also raises downside beta from {without.downside_beta:.2f} to {with_c.downside_beta:.2f}."
        ),
        text,
    )
    if n != 1:
        raise ValueError("Missing or ambiguous abstract selection sentence")
    text = _paragraph(
        text,
        "The net annualized mean-return gain from adding",
        rf"The net annualized mean-return gain from adding $C$ is {full.mean_diff:.2f} percentage points ($t={full.hac_t:.2f}$); the paired Sharpe gain is {full.delta_sr:.3f} (Panel B). The later-period interval includes zero, as does the interval for the change in the Sharpe gain between periods: ${change['sr_change']:.3f}$ $[{lower:.3f},{upper:.3f}]$. The annualized mean gain falls by {abs(change['mean_diff']):.2f} percentage points ($t={change['t']:.2f}$). We find a full-sample increment but cannot establish that it changed after 2011 or persists outside this sample. Signal comparisons use reconstructed returns. Market comparisons use the original allocation returns; the two series are kept separate.",
    )
    contrasts = table("removal_contrasts", ["period", "removed_1", "removed_2"]).loc["Full"]
    significant = [k for k in "BVC" if removal.loc[k, "lo"] > 0]
    precise = (
        "every pairwise contrast between the three effects includes zero"
        if ((contrasts.lo <= 0) & (contrasts.hi >= 0)).all()
        else "some pairwise intervals exclude zero"
    )
    text = _paragraph(
        text,
        "Removing $B$, $V$, or $C$",
        rf"Removing $B$, $V$, or $C$ reduces full-period Sharpe by {removal.loc['B', 'delta']:.3f}, {removal.loc['V', 'delta']:.3f}, or {removal.loc['C', 'delta']:.3f}, respectively. Paired 95\% intervals for the removal effects exclude zero for "
        + " and ".join("$" + k + "$" for k in significant)
        + f", but {precise}. "
        + rf"Monthly Fama--MacBeth cross-sectional regressions of observed next-month adjusted-price returns on all three percentile scores \cite{{famamacbeth}} yield a mean monthly slope of {fmb.loc[('Full', 'C'), 'slope_monthly_pct']:.2f} percentage points per unit of the fractional rank of $C$ ($t={fmb.loc[('Full', 'C'), 'hac_t']:.2f}$). The $B$ and $V$ slopes are ${fmb.loc[('Full', 'B'), 'slope_monthly_pct']:.2f}$ and ${fmb.loc[('Full', 'V'), 'slope_monthly_pct']:.2f}$ points; neither is significant. The $C$ slope is {fmb.loc[('2012--20', 'C'), 'slope_monthly_pct']:.2f} points ($t={fmb.loc[('2012--20', 'C'), 'hac_t']:.2f}$) in 2012--2020. These unweighted cross-sectional regressions use untrimmed observed returns and omit missing dependent observations. A slope measures return predictability; unlike the portfolio Sharpe comparison, it does not reward a reduction in risk.",
    )
    text = _paragraph(
        text,
        "Adding $C$ raises downside beta",
        rf"Adding $C$ raises downside beta, estimated by regressing strategy excess returns on the FF excess-market return in months when that market excess return is negative, from {without.downside_beta:.2f} to {with_c.downside_beta:.2f}. Ranking months by the FF total-market return, the mean portfolio return in the {int(with_c.bottom_decile_n)} bottom-decile months falls from ${without.bottom_decile_mean:.2f}\%$ to ${with_c.bottom_decile_mean:.2f}\%$; in the five worst months it falls from ${without.worst_five_mean:.2f}\%$ to ${with_c.worst_five_mean:.2f}\%$. A shallower full-period MDD therefore does not imply better crash protection. In the seven worst market months of Crisis I, $B+V+C$ loses {abs(with_c.crisis_worst_seven):.2f}\% cumulatively versus {abs(without.crisis_worst_seven):.2f}\% for $B+V$; over the remaining 14 months it gains {with_c.crisis_other_fourteen:.2f}\% versus {without.crisis_other_fourteen:.2f}\%. The advantage comes from returns outside the worst months.",
    )
    text = _paragraph(
        text,
        "Across eight specifications varying",
        rf"Across eight specifications varying $K$, incumbent retention, or the industry count limit, the full-period Sharpe increment from $C$ ranges from {sensitivity.delta_sr.min():.3f} to {sensitivity.delta_sr.max():.3f}. The gain is {sensitivity.loc['K30', 'delta_sr']:.3f} at $K=30$ but only {sensitivity.loc['K100', 'delta_sr']:.3f} at $K=100$. For $K=40$, 50, and 100 the full-period MDD deepens rather than improves. The sign of the Sharpe increment is stable across these specifications; its size and drawdown consequences are not. These comparisons use the same historical sample and are not independent validations.",
    )
    precision = table("bootstrap_precision", "model")
    names = {
        "BVCM": "12--1 momentum",
        "BVCL": "distance from the 52-week low",
        "BVCP": "profitability beta",
        "BVCI": "net issuance",
    }
    negative = [names[k] for k in names if adds.loc[k, "ci_hi"] < 0]
    text = _paragraph(
        text,
        "To test a change in the signal rule itself",
        r"To test a change in the signal rule itself, we keep 30 stocks, the retention rule, the industry limit and the position cap fixed. We add one return signal at a time to the three defensive scores. All four versions have lower estimated CAGR and Sharpe than the original rule (Table~\ref{tab:selectionrisk}). In the seeded replication the pointwise Sharpe intervals exclude zero for "
        + " and ".join(negative)
        + rf". Repeating the paired calculation with {int(precision.loc['BVCP', 'draws']):,} draws gives an upper endpoint of ${precision.loc['BVCP', 'ci_hi']:+.3f}$ for profitability beta and ${precision.loc['BVCL', 'ci_hi']:+.3f}$ for the 52-week-low signal. The profitability result is therefore sensitive to Monte Carlo variation at this cutoff; none of the intervals adjusts for multiple comparisons. A separate test mixes the monthly net returns of the defensive and return-oriented sleeves. At 80/20, CAGR rises by {a.cagr - base.cagr:.2f} points and the estimated Sharpe difference is {a.delta_sr:.3f}, but drawdown deepens by {base.mdd - a.mdd:.2f} points. An interval containing zero does not establish equal Sharpe ratios.",
    )
    text = _paragraph(
        text,
        "The selection rerun gives a",
        rf"The selection rerun gives a {adds.loc['RO', 'cagr']:.2f}\% stand-alone CAGR for the 50-stock portfolio rather than the 16.48\% obtained from the recorded weights and article accounting. The blend comparisons use the rerun throughout; the two return series are not pooled.",
    )
    t = tmp.loc[4]
    rr = roll.loc[60]
    r36 = roll.loc[36]
    text = _paragraph(
        text,
        "$12\\delta$ is the change in annualized alpha",
        rf"$12\delta$ is the change in annualized alpha and $\gamma$ the change in factor loadings. Alpha fell by {abs(t.alpha_change):.2f} percentage points ($t={t.alpha_change_t:.2f}$, HAC(4)). Market and investment-factor (CMA) loadings rose by {t.market_change:.2f} and {t.cma_change:.2f}. A joint Wald test of $\gamma=0$ gives $p={tmp.loc[4, 'wald_p']:.3f}$, {tmp.loc[6, 'wald_p']:.3f} and {tmp.loc[12, 'wald_p']:.3f} at HAC lags 4, 6 and 12. The intercept declined, but the evidence for changing exposures depends on the HAC lag. These regressions do not identify why the intercept fell. Rolling 60-month FF5 alpha was {rr.loc['2008-12', 'alpha']:.2f}\% in December 2008, {rr.loc['2012-12', 'alpha']:.2f}\% in December 2012 and {rr.loc['2020-12', 'alpha']:.2f}\% in December 2020; the market loading ranged from {rr.market_beta.min():.2f} to {rr.market_beta.max():.2f} ({r36.market_beta.min():.2f} to {r36.market_beta.max():.2f} in 36-month regressions). Overlapping windows are descriptive, not independent tests.",
    )
    b = reg.loc["original_ff5_bab"]
    low = reg.loc["rerun_ff5_bab_lowvol"]
    cs = reg.loc["coskew_ff5_bab"]
    text = _paragraph(
        text,
        r"Table~\ref{tab:inference} uses the original portfolio returns throughout.",
        rf"Table~\ref{{tab:inference}} uses the original portfolio returns throughout. In the FF5+BAB regression the BAB loading is {b.loc['Full', 'BAB']:.3f} ($t={b.loc['Full', 'BAB_t']:.2f}$) over the full sample and {b.loc['2012--20', 'BAB']:.3f} ($t={b.loc['2012--20', 'BAB_t']:.2f}$) after 2011. Separately, tests based on the reconstructed selection series add a CRSP low-volatility long--short control to FF5+BAB. The full-sample strategy alpha is {low.loc['Full', 'alpha']:.2f}\% ($t={low.loc['Full', 'alpha_t']:.2f}$), versus {low.loc['2012--20', 'alpha']:.2f}\% ($t={low.loc['2012--20', 'alpha_t']:.2f}$) after 2011. A capitalization-weighted long--short portfolio of the top versus bottom 30\% of the coskewness score has FF5+BAB alpha of {cs.loc['Full', 'alpha']:.2f}\% ($t={cs.loc['Full', 'alpha_t']:.2f}$) over the full period and {cs.loc['2012--20', 'alpha']:.2f}\% ($t={cs.loc['2012--20', 'alpha_t']:.2f}$) after 2011. Both long--short controls use gross, uncapped value-weighted legs and within-universe percentile scores. They are factor diagnostics, not the investable long-only strategy.",
    )
    text = _paragraph(
        text,
        "Across the earlier and later periods",
        rf"Across the earlier and later periods, annualized mean BAB rose from {means.loc['2003--11', 'bab']:.2f}\% to {means.loc['2012--20', 'bab']:.2f}\%, while the coskewness spread fell from {means.loc['2003--11', 'coskew']:.2f}\% to {means.loc['2012--20', 'coskew']:.2f}\%. A weaker realized BAB return therefore cannot account for the narrower later Sharpe advantage. The change in the estimated overall FF5 intercept is distinct from a change in the incremental effect of adding coskewness to the selection rule; the paired signal comparison does not establish that the latter change is significant.",
    )
    caps = table("capture_windows", ["strategy", "period"])
    pre = json.loads((revisions / "precrisis_beta.json").read_text())
    crisis = table("crisis_contribution", "market")
    o = caps.loc["original"]
    v = caps.loc["BV"]
    cw = crisis.loc["CRSP_market"]
    ff = crisis.loc["FF_market"]
    text = _paragraph(
        text,
        "We separate four peak-to-trough or rebound windows",
        rf"We separate four peak-to-trough or rebound windows from the nine calendar regimes; the March--May 2009 rebound, for example, crosses the Crisis I/Recovery I boundary. Capture here is the portfolio’s cumulative return divided by the concurrent \emph{{FF total-market}} cumulative return. It is a return ratio, not an estimated beta. Capped VW absorbed {o.loc['2007_decline', 'capture_pct']:.0f}\% of the November 2007--February 2009 market decline and {o.loc['2009_rebound', 'capture_pct']:.0f}\% of the March--May 2009 rebound. It absorbed {o.loc['2020_decline', 'capture_pct']:.0f}\% of the February--March 2020 decline but only {o.loc['2020_rebound', 'capture_pct']:.0f}\% of the April--June rebound. In 2020, the strategy therefore entered the rebound with almost the market’s loss and recovered less than half its gain.",
    )
    text = _paragraph(
        text,
        "The matched selection comparison helps distinguish",
        rf"The matched selection comparison helps distinguish this result from the choice of weights: without coskewness, the low-beta/low-volatility portfolio captured {v.loc['2020_decline', 'capture_pct']:.0f}\% of the 2020 decline and {v.loc['2020_rebound', 'capture_pct']:.0f}\% of the rebound. The January 2020 holdings had an estimated FF5 market loading of {pre['BV']['market_beta']:.2f} without coskewness and {pre['BVC']['market_beta']:.2f} with it. These estimates weight individual stock slopes by current holdings using available observations in the preceding 36 months ({pre['BV']['min_observations']}--36 observations without coskewness, {pre['BVC']['min_observations']} with it). The first decline spanned {int(o.loc['2007_decline', 'months'])} monthly observations, the second only {int(o.loc['2020_decline', 'months'])}. These two episodes do not identify the effect of rebalancing speed or policy intervention. The detailed industry-versus-stock attribution could not be independently reproduced; we do not assign the shortfall to particular industries.",
    )
    text = _paragraph(
        text,
        "Crisis I contributed",
        rf"Crisis I contributed {cw.crisis_log_points:.1f} of the {cw.total_log_points:.1f} log-return points by which Capped VW outgrew CRSP over the full sample, where one point is $100$ times a difference in log wealth. Against FF, Crisis I contributed {ff.crisis_log_points:.1f} of {ff.total_log_points:.1f} points. Omitting the crisis leaves positive estimated Sharpe differences over the remaining {int(ff.other_months)} months: {cw.other_sharpe_difference:.3f} versus CRSP and {ff.other_sharpe_difference:.3f} versus FF. With paired six-month blocks drawn separately within the two surviving segments and each segment’s length held fixed, the 95\% intervals are $[{cw.ci_lo:.3f},{cw.ci_hi:.3f}]$ and $[{ff.ci_lo:.3f},{ff.ci_hi:.3f}]$, respectively. Much of the cumulative advantage accrued during the financial crisis. Neither result establishes persistence beyond this sample.",
    )
    return text


if __name__ == "__main__":
    main()
