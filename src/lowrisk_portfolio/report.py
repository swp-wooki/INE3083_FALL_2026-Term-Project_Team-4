"""Refresh reviewed article tables and inference from current experiment results."""

from .paths import OUTPUT, PAPER, ROOT

P = OUTPUT
OUT = OUTPUT
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import performance

M = pd.read_csv(P / "model_metrics.csv")
R = pd.read_csv(P / "monthly_returns.csv")
Q = pd.read_csv(P / "paired_comparisons.csv")
B = pd.read_csv(P / "benchmark_returns.csv", index_col=0)
BM = pd.read_csv(P / "benchmark_metrics.csv")
A = pd.read_csv(P / "data_audit.csv")
S = pd.read_csv(P / "frozen_selection.csv")
PRIMARY = "total_observed_drift10"
MARKETS = ["SP500", "CRSP_market", "FF_market"]
MARKET_NAMES = {
    "SP500": r"S\&P 500 (price)",
    "CRSP_market": "CRSP market",
    "FF_market": "FF market",
}
DISPLAY = ["source_weight", "equal_weight", "gmvp_constant_corr", *MARKETS]
full = M.query('scenario == @PRIMARY and period == "full"').set_index("model")
expected_scenarios = {f"total_observed_drift{c}" for c in [0, 10, 25, 50, 75, 100]}
assert set(R.scenario) == expected_scenarios and set(M.scenario) == expected_scenarios
assert len(full) == 19 and full.index.is_unique
names = {
    "source_weight": "Market-cap weight",
    "equal_weight": "Equal weight",
    "gmvp_sample": "Sample GMVP",
    "gmvp_constant_corr": "Constant-corr. GMVP",
    "gmvp_lw_constant_corr": "LW constant-corr.",
    "gmvp_lw_identity": "LW identity",
    "gmvp_rmt_clipping": "RMT clipping",
    "gmvp_qis": "QIS",
    "gmvp_ridge": "Ridge GMVP",
    "gmvp_factor": "FF5 GMVP",
    "inverse_vol": "Inverse volatility",
    "erc": "Constrained risk budgeting",
    "mdp": "Maximum diversification",
    "max_sharpe_bs": "Bayes--Stein Sharpe",
    "min_cvar": "Minimum CVaR",
    "mix_1n": "GMVP--1/N mixture",
    "norm_gmvp": "Norm GMVP",
    "tc_gmvp": "Transaction-cost GMVP",
    "vol_timing": "Inverse variance",
}


def allocation_label(model):
    label = MARKET_NAMES.get(model, names.get(model, model.replace("_", " ")))
    return r"\hyperref[alloc:" + model + "]{" + label + "}"


assert np.isfinite(full[["cagr", "sharpe", "annual_volatility", "max_drawdown"]].to_numpy()).all()
# Paired circular block bootstrap: identical resampled months across all models and RF.
wide = R.query("scenario == @PRIMARY").pivot(index="month", columns="model", values="net_return")
assert wide.shape == (215, 19) and wide.index.is_unique
excess = wide.to_numpy() - B.loc[wide.index, "RF"].to_numpy()[:, None]
rng = np.random.default_rng(20261002)
T = len(wide)
nboot = 2000
block = 6
sr = excess.mean(0) / excess.std(0, ddof=1) * np.sqrt(12)
samples = []
for _ in range(nboot):
    starts = rng.integers(0, T, size=(T + block - 1) // block)
    idx = ((starts[:, None] + np.arange(block)) % T).ravel()[:T]
    e = excess[idx]
    samples.append(e.mean(0) / e.std(0, ddof=1) * np.sqrt(12))
samples = np.array(samples)
boot = []
for base in ["source_weight", "equal_weight"]:
    j = wide.columns.get_loc(base)
    for i, model in enumerate(wide.columns):
        if model == base:
            continue
        lo, hi = np.quantile(samples[:, i] - samples[:, j], [0.025, 0.975])
        boot.append(
            dict(
                model=model,
                baseline=base,
                delta_sharpe=sr[i] - sr[j],
                ci_low=lo,
                ci_high=hi,
                replications=nboot,
                block_months=block,
                seed=20261002,
            )
        )
pd.DataFrame(boot).to_csv(P / "sharpe_bootstrap.csv", index=False)
# The required S&P benchmark is a price-return series, never relabeled total return.
bench = pd.read_csv(ROOT / "data/raw/market_benchmarks.csv").set_index("month")
assert bench.index.is_unique and B.index.is_unique
sp = bench.loc[B.index, "sp500_return"].copy()
assert np.isfinite(sp).all() and sp.gt(-1).all() and len(sp) == 215
B["SP500"] = sp
periods = {
    "full": ("2003-02", "2020-12"),
    "2003_2011": ("2003-02", "2011-12"),
    "2012_2020": ("2012-01", "2020-12"),
    "financial_crisis": ("2007-07", "2009-03"),
    "covid": ("2020-02", "2020-06"),
}
BM = pd.concat(
    [
        BM.loc[BM.model.ne("SP500")],
        pd.DataFrame(
            [
                dict(model="SP500", period=period, **performance(sp.loc[a:b], B.RF))
                for period, (a, b) in periods.items()
            ]
        ),
    ],
    ignore_index=True,
)
B.rename_axis("month").to_csv(P / "benchmark_returns.csv")
BM.to_csv(P / "benchmark_metrics.csv", index=False)
sp.index = pd.PeriodIndex(sp.index, freq="M")
rf = B.RF.copy()
rf.index = pd.PeriodIndex(rf.index, freq="M")
spstats = performance(sp, rf)
pd.DataFrame([spstats]).to_csv(P / "sp500_price_metrics.csv", index=False)
selection_audit = dict(
    score_max_mean_error=float(
        (S[["factor_1", "factor_2", "factor_3"]].mean(axis=1) - S.composite_score).abs().max()
    ),
    formation_size_missing=int(A["size"].isna().sum()),
    formation_sic_missing=int(A.sic.isna().sum()),
    observed_max_industry_count=int(
        A.assign(ind=A.sic // 100).groupby(["holding_month", "ind"]).size().max()
    ),
    metrics_finite=True,
    bootstrap_seed=20261002,
    bootstrap_replications=nboot,
)
(P / "supplementary_audit.json").write_text(json.dumps(selection_audit, indent=2))


def pct(x):
    return f"{x * 100:.2f}"


def row(values):
    return " & ".join(str(v) for v in values) + r" \\" + "\n"


def get(model, scenario=PRIMARY, period="full"):
    return M.query("model == @model and scenario == @scenario and period == @period").iloc[0]


main = ""
for model, v in full.sort_values("sharpe", ascending=False).iterrows():
    main += row(
        [
            allocation_label(model),
            pct(v.cagr),
            pct(v.annual_volatility),
            f"{v.sharpe:.3f}",
            pct(v.max_drawdown),
            pct(v.mean_turnover),
        ]
    )
for model in MARKETS:
    v = BM.query('model == @model and period == "full"').iloc[0]
    main += row(
        [
            allocation_label(model),
            pct(v.cagr),
            pct(v.annual_volatility),
            f"{v.sharpe:.3f}",
            pct(v.max_drawdown),
            "--",
        ]
    )
# User-supplied regime calendar; February 2020 belongs only to Crisis II.
regimes = [
    ("Bull I", "Bull", "2003-01", "2007-06"),
    ("Crisis I", "Crisis", "2007-07", "2009-03"),
    ("Recovery I", "Recovery", "2009-04", "2012-06"),
    ("Bull II", "Bull", "2012-07", "2015-06"),
    ("Shock I", "Shock", "2015-07", "2016-03"),
    ("Bull III", "Bull", "2016-04", "2018-12"),
    ("Pre-Crisis", "Bull", "2019-01", "2020-01"),
    ("Crisis II", "Crisis", "2020-02", "2020-06"),
    ("Recovery II", "Recovery", "2020-07", "2021-12"),
    ("Tightening", "Shock", "2022-01", "2022-12"),
    ("Stabilization", "Stable", "2023-01", "2024-12"),
    ("Pre-Tariff", "Shock", "2025-01", "2025-03"),
    ("Tariff Shock", "Shock", "2025-04", "2025-06"),
    ("Post-Tariff", "Stable", "2025-07", "2025-12"),
]
coverage = []
regime_results = []
assigned = []
for label, kind, start, end in regimes:
    months = wide.loc[start:end].index
    if not len(months):
        continue
    status = "Full" if months[0] == start and months[-1] == end else "Partial"
    coverage.append(
        dict(
            regime=label,
            type=kind,
            requested_start=start,
            requested_end=end,
            observed_start=months[0] if len(months) else "",
            observed_end=months[-1] if len(months) else "",
            months=len(months),
            coverage=status,
        )
    )
    assigned.extend(months)
    if not len(months):
        continue
    for (scenario, model), g in R.groupby(["scenario", "model"], sort=False):
        series = g.set_index("month").net_return.loc[months]
        regime_results.append(
            dict(
                regime=label,
                type=kind,
                scenario=scenario,
                model=model,
                start=months[0],
                end=months[-1],
                **performance(series, B.RF),
            )
        )
    for model in MARKETS:
        regime_results.append(
            dict(
                regime=label,
                type=kind,
                scenario="benchmark",
                model=model,
                start=months[0],
                end=months[-1],
                **performance(B.loc[months, model], B.RF),
            )
        )
assert len(assigned) == len(set(assigned)) == len(wide) and set(assigned) == set(wide.index)
C = pd.DataFrame(coverage)
G = pd.DataFrame(regime_results)
assert G.groupby(["scenario", "model"]).months.sum().eq(len(wide)).all()
# Regime slicing must preserve each full-period compounded path exactly.
for (scenario, model), g in G.groupby(["scenario", "model"]):
    expected = (
        BM.query('model == @model and period == "full"')
        if scenario == "benchmark"
        else M.query('scenario == @scenario and model == @model and period == "full"')
    ).iloc[0]
    assert np.isclose((1 + g.cumulative_return).prod() - 1, expected.cumulative_return, atol=1e-12)
assert np.isfinite(G[["cagr", "annual_volatility", "sharpe", "max_drawdown"]].to_numpy()).all()
C.to_csv(P / "regime_coverage.csv", index=False)
G.to_csv(P / "regime_metrics.csv", index=False)
coverage_table = ""
sub = ""
regime_blocks = []
for v in coverage:
    requested = v["requested_start"] + "--" + v["requested_end"]
    observed = (v["observed_start"] + "--" + v["observed_end"]) if v["months"] else "--"
    coverage_table += row([v["regime"], v["type"], requested, observed, v["months"], v["coverage"]])
    if not v["months"]:
        continue
    for j, model in enumerate(DISPLAY):
        z = G[
            (G.regime == v["regime"]) & (G.model == model) & G.scenario.isin([PRIMARY, "benchmark"])
        ].iloc[0]
        sub += row(
            [
                v["regime"] if j == 0 else "",
                MARKET_NAMES.get(model, names.get(model, model)),
                pct(z.cumulative_return),
                pct(z.cagr),
                pct(z.annual_volatility),
                f"{z.sharpe:.3f}",
                pct(z.max_drawdown),
            ]
        )
    sub += "\\addlinespace[2pt]\n"


def regime_value(label, model, metric):
    return G[
        (G.regime == label) & (G.model == model) & G.scenario.isin([PRIMARY, "benchmark"])
    ].iloc[0][metric]


def regime_pct(label, model, metric):
    return pct(regime_value(label, model, metric)) + r"\%"


def refreshed_regime_paragraphs(results):
    selected = results[results.scenario.isin([PRIMARY, "benchmark"])]

    def val(regime, model, key):
        return float(selected[(selected.regime == regime) & (selected.model == model)].iloc[0][key])

    def per(regime, model, key="cumulative_return"):
        return f"{100 * val(regime, model, key):.2f}" + r"\%"

    def sharpe(regime, model):
        return f"{val(regime, model, 'sharpe'):.3f}"

    def strategies(regime):
        return (
            selected[(selected.regime == regime) & selected.scenario.eq(PRIMARY)]
            .set_index("model")
            .cumulative_return
        )

    def difference(regime):
        return 100 * (
            val(regime, "gmvp_constant_corr", "cumulative_return")
            - val(regime, "source_weight", "cumulative_return")
        )

    def comparison_count(regime):
        x = strategies(regime)
        market = val(regime, "CRSP_market", "cumulative_return")
        return f"{int((x > market).sum())} of {len(x)}"

    def return_range(regime):
        x = strategies(regime)
        return f"${100 * x.min():.2f}" + r"\%$ to $" + f"{100 * x.max():.2f}" + r"\%$"

    def winners(regime):
        x = selected[(selected.regime == regime) & selected.model.isin(DISPLAY)].set_index("model")
        winner_names = {**names, "CRSP_market": "CRSP market"}
        a = x.cumulative_return.idxmax()
        b = x.sharpe.idxmax()
        return (
            regime
            + ": "
            + winner_names[a]
            + " has the highest cumulative return ("
            + per(regime, a)
            + "), and "
            + winner_names[b]
            + " has the highest Sharpe ratio ("
            + sharpe(regime, b)
            + ")"
        )

    p = {}
    p["During Crisis I,"] = (
        "During Crisis I, market-capitalization weighting produces a cumulative return of $"
        + per("Crisis I", "source_weight")
        + "$, compared with $"
        + per("Crisis I", "CRSP_market")
        + "$ for the market. "
        "The constant-correlation GMVP records annualized volatility of "
        + per("Crisis I", "gmvp_constant_corr", "annual_volatility")
        + ", compared with "
        + per("Crisis I", "source_weight", "annual_volatility")
        + " for market-capitalization weighting, "
        "and a within-regime MDD of $"
        + per("Crisis I", "gmvp_constant_corr", "max_drawdown")
        + "$. Its cumulative return of $"
        + per("Crisis I", "gmvp_constant_corr")
        + r"$ is underlined in Table~\ref{tab:period}. "
        "These separate comparisons show why cumulative return, volatility, and drawdown must be considered jointly. "
        "When average excess returns are negative, a smaller volatility denominator can make the Sharpe ratio more negative; the ratio alone does not order both return and risk."
    )
    p["Relative performance varies across regimes."] = (
        "Relative performance varies across regimes. In Recovery I, constant correlation produces a cumulative return of "
        + per("Recovery I", "gmvp_constant_corr")
        + ", compared with "
        + per("Recovery I", "source_weight")
        + " for market-capitalization weighting. "
        "In Bull II, the respective returns are "
        + per("Bull II", "gmvp_constant_corr")
        + " and "
        + per("Bull II", "source_weight")
        + ", compared with "
        + per("Bull II", "CRSP_market")
        + " for the market. In Shock I, market-capitalization weighting and constant correlation return "
        + per("Shock I", "source_weight")
        + " and "
        + per("Shock I", "gmvp_constant_corr")
        + ", respectively, while the market returns $"
        + per("Shock I", "CRSP_market")
        + "$. Among the four displayed portfolios, the bold entries identify the following descriptive maxima: "
        + "; ".join(winners(r) for r in ["Bull I", "Bull III", "Pre-Crisis"])
        + ". These rankings document changes in relative performance without identifying the causal contribution of individual selection characteristics."
    )
    p["The Crisis II results"] = (
        "The Crisis II results do not support a general claim of superior crisis-period performance. From February through June 2020, "
        "market-capitalization weighting and constant correlation return $"
        + per("Crisis II", "source_weight")
        + "$ and $"
        + per("Crisis II", "gmvp_constant_corr")
        + "$, respectively, compared with $"
        + per("Crisis II", "CRSP_market")
        + "$ for the market. "
        "The underlined constant-correlation return must be read together with its volatility and recovery path; the market Sharpe ratio is "
        + sharpe("Crisis II", "CRSP_market")
        + ". During the observed July--December 2020 portion of Recovery II, "
        "market-capitalization weighting returns "
        + per("Recovery II", "source_weight")
        + ", constant correlation returns "
        + per("Recovery II", "gmvp_constant_corr")
        + ", and the market returns "
        + per("Recovery II", "CRSP_market")
        + ". Their Sharpe ratios are "
        + sharpe("Recovery II", "source_weight")
        + ", "
        + sharpe("Recovery II", "gmvp_constant_corr")
        + ", and "
        + sharpe("Recovery II", "CRSP_market")
        + ", respectively. The bold table entries distinguish the highest cumulative growth "
        "from the highest volatility-adjusted performance over this short recovery interval."
    )
    p["The complete comparison reveals"] = (
        "The complete comparison reveals a distinction that is obscured by a single full-sample ranking. In Crisis I, "
        + comparison_count("Crisis I")
        + " selected-stock allocations outperform the CRSP market cumulatively: strategy returns range from "
        + return_range("Crisis I")
        + ", compared with $"
        + per("Crisis I", "CRSP_market")
        + "$ for the market. In Crisis II, "
        + comparison_count("Crisis II")
        + " outperform: strategy returns range from "
        + return_range("Crisis II")
        + ", compared with $"
        + per("Crisis II", "CRSP_market")
        + "$ for the market. These statements use observed security returns and the original round-trip cost "
        "of 20 basis points. They are conditional on the specified crisis windows, including the April--June 2020 rebound, and do not establish "
        "a general crisis-protection property of low-risk selection."
    )
    order = list(dict.fromkeys(selected.regime))
    positive = [r for r in order if difference(r) > 0]
    negative = [r for r in order if difference(r) < 0]

    def listed(regimes):
        return ", ".join(r + " ($" + f"{difference(r):+.2f}" + "$)" for r in regimes)

    below = strategies("Recovery II").drop("source_weight") < val(
        "Recovery II", "source_weight", "cumulative_return"
    )
    p["The heatmap also shows"] = (
        "The heatmap also shows that constant-correlation GMVP's cumulative-return difference from market-capitalization weighting is positive in "
        + listed(positive)
        + " and negative in "
        + listed(negative)
        + ". All parenthetical differences are percentage points. "
        "In the observed portion of Recovery II, "
        + str(int(below.sum()))
        + " of "
        + str(len(below))
        + " alternative allocations trail market-capitalization weighting. "
        "These within-regime contrasts demonstrate variation in the incremental outcome of reweighting. Their magnitudes should not be compared "
        "across unequal-length intervals as return rates, and the colors do not denote statistical significance."
    )
    return p


costs = ""
for c in [0, 10, 25, 50, 75, 100]:
    vals = [
        get(model, f"total_observed_drift{c}")
        for model in ["source_weight", "equal_weight", "gmvp_constant_corr", "tc_gmvp"]
    ]
    costs += row([c, *[pct(v.cagr) for v in vals]])
inference = ""
for model in [
    "equal_weight",
    "gmvp_constant_corr",
    "tc_gmvp",
    "inverse_vol",
    "gmvp_sample",
    "max_sharpe_bs",
]:
    v = Q.query(
        'scenario == @PRIMARY and period == "full" and baseline == "source_weight" and model == @model'
    ).iloc[0]
    b = next(b for b in boot if b["model"] == model and b["baseline"] == "source_weight")
    inference += row(
        [
            allocation_label(model),
            pct(v.annualized_difference),
            f"{v.hac_t:.2f}",
            f"{v.holm_p:.3f}",
            f"{b['delta_sharpe']:.3f}",
            f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]",
        ]
    )


# Self-contained PGFPlots: no external figure files are required by the native editor.
def plot(kind):
    styles = [
        ("source_weight", "black,thick"),
        ("equal_weight", "gray,dashed"),
        ("gmvp_constant_corr", "blue,thick"),
    ]
    out = ""
    for name, style in styles + [("CRSP_market", "red,densely dotted")]:
        ret = B.CRSP_market.to_numpy() if name == "CRSP_market" else wide[name].to_numpy()
        wealth = np.r_[1, np.cumprod(1 + ret)]
        y = wealth if kind == "wealth" else (wealth / np.maximum.accumulate(wealth) - 1) * 100
        coords = " ".join(f"({i},{v:.5f})" for i, v in enumerate(y))
        out += r"\addplot[" + style + "] coordinates {" + coords + "};\n"
        out += (
            r"\addlegendentry{" + ("CRSP market" if name == "CRSP_market" else names[name]) + "}\n"
        )
    return out


D = pd.read_csv(P / "mechanism_summary.csv").set_index("model")
structure = pd.read_csv(P / "selected_covariance_structure.csv")
mechanism_table = ""
for model in [
    "source_weight",
    "equal_weight",
    "gmvp_constant_corr",
    "gmvp_lw_constant_corr",
    "gmvp_sample",
    "gmvp_lw_identity",
    "gmvp_qis",
    "gmvp_ridge",
    "gmvp_rmt_clipping",
    "gmvp_factor",
    "inverse_vol",
    "vol_timing",
    "erc",
    "mix_1n",
    "norm_gmvp",
    "tc_gmvp",
    "min_cvar",
    "mdp",
    "max_sharpe_bs",
]:
    v = D.loc[model]
    mechanism_table += row(
        [
            allocation_label(model),
            f"{v.effective_n:.1f}",
            f"{v.cap_count:.1f}",
            pct(v.tilt_score1),
            pct(v.tilt_score2),
            pct(v.tilt_score3),
            pct(v.delta_annual_gross_mean),
            pct(v.delta_annual_cost_drag),
            pct(v.delta_annual_net_mean),
        ]
    )

# Keep reviewed prose in the open document; update only named data elements.
import re
from subprocess import run as subprocess_run
from tempfile import TemporaryDirectory

article = P / "paper/main.tex"
article.parent.mkdir(parents=True, exist_ok=True)
text = (PAPER / "main.tex").read_text()


def emph(value, kind="textbf"):
    return "\\" + kind + "{" + value + "}"


def replace_rows(label, rows):
    global text
    a = text.index("\\label{" + label + "}")
    start = text.index("\\midrule", a) + len("\\midrule")
    end = text.index("\\bottomrule", start)
    text = text[:start] + "\n" + rows + text[end:]


def replace_paragraph(prefix, replacement):
    global text
    aliases = {
        "The estimated maximum-Sharpe portfolio has": "The revised estimated maximum-Sharpe portfolio has",
        "ERC has an effective number": "Constrained risk budgeting has an average effective number",
        "Maximum diversification reduces": "Maximum diversification changes",
        "Additional files contain characteristic-exposure measures,": "Additional files contain characteristic exposures,",
    }
    search_from = (
        text.index(r"\appendices")
        if prefix.startswith("Additional files")
        else text.index(r"\section{Portfolio Performance}")
    )
    needle = prefix if prefix in text[search_from:] else aliases[prefix]
    start = text.index(needle, search_from)
    end = text.find("\n\n", start)
    if end < 0:
        raise ValueError(prefix)
    text = text[:start] + replacement + text[end:]


main = ""
fields = ["cagr", "annual_volatility", "sharpe", "max_drawdown", "mean_turnover"]
favorable = {
    k: (full[k].min() if k in ["annual_volatility", "mean_turnover"] else full[k].max())
    for k in fields
}
for model, v in full.sort_values("sharpe", ascending=False).iterrows():
    vals = []
    for key in fields:
        val = f"{v[key]:.3f}" if key == "sharpe" else pct(v[key])
        if np.isclose(v[key], favorable[key], atol=1e-12, rtol=0):
            val = emph(val)
        if (model == "source_weight" and key in ["cagr", "sharpe"]) or (
            key == "sharpe" and v[key] == full.sharpe.min()
        ):
            val = emph(val, "underline")
        vals.append(val)
    main += row([allocation_label(model), *vals])
for model in MARKETS:
    v = BM.query('model == @model and period == "full"').iloc[0]
    vals = [
        pct(v.cagr),
        pct(v.annual_volatility),
        f"{v.sharpe:.3f}",
        pct(v.max_drawdown),
    ]
    if model == "SP500":
        vals = [emph(x, "underline") for x in vals]
    main += row(
        [
            emph(allocation_label(model)) if model == "SP500" else allocation_label(model),
            *vals,
            "--",
        ]
    )
replace_rows("tab:main", main)

inference = ""
for model in [
    "equal_weight",
    "gmvp_constant_corr",
    "tc_gmvp",
    "inverse_vol",
    "gmvp_sample",
    "max_sharpe_bs",
]:
    v = Q.query(
        'scenario == @PRIMARY and period == "full" and baseline == "source_weight" and model == @model'
    ).iloc[0]
    b = next(b for b in boot if b["model"] == model and b["baseline"] == "source_weight")
    vals = [
        pct(v.annualized_difference),
        f"{v.hac_t:.2f}",
        f"{v.holm_p:.3f}",
        f"{b['delta_sharpe']:.3f}",
        f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]",
    ]
    if v.holm_p < 0.05:
        vals[0] = emph(vals[0])
        vals[2] = emph(vals[2])
    if model == "gmvp_constant_corr":
        vals[2] = emph(vals[2], "underline")
        vals[4] = emph(vals[4], "underline")
    inference += row([allocation_label(model), *vals])
replace_rows("tab:inference", inference)

coverage_table = ""
sub = ""
for v in coverage:
    observed = v["observed_start"] + "--" + v["observed_end"]
    count = str(v["months"])
    if v["regime"] in ["Pre-Crisis", "Crisis II"]:
        observed = emph(observed)
    if v["months"] < 12:
        count = emph(count, "underline") if v["regime"] in ["Crisis II", "Recovery II"] else count
    coverage_table += row([v["regime"], v["type"], observed, count])
    subset = G[
        (G.regime == v["regime"]) & G.model.isin(DISPLAY) & G.scenario.isin([PRIMARY, "benchmark"])
    ].set_index("model")
    block = r"\multicolumn{6}{@{}l}{\textit{" + v["regime"] + r"}}\\[1pt]" + "\n"
    for j, model in enumerate(DISPLAY):
        z = subset.loc[model]
        vals = [
            pct(z.cumulative_return),
            pct(z.cagr),
            pct(z.annual_volatility),
            f"{z.sharpe:.3f}",
            pct(z.max_drawdown),
        ]
        if (
            model not in MARKETS
            and z.cumulative_return
            == subset.loc[[m for m in DISPLAY if m not in MARKETS], "cumulative_return"].max()
        ):
            vals[0] = emph(vals[0])
        if (
            model not in MARKETS
            and z.sharpe == subset.loc[[m for m in DISPLAY if m not in MARKETS], "sharpe"].max()
        ):
            vals[3] = emph(vals[3])
        if model == "gmvp_constant_corr" and v["regime"] in ["Crisis I", "Crisis II"]:
            vals[0] = emph(vals[0], "underline")
        if model == "SP500":
            vals = [emph(x, "underline") for x in vals]
        sub += row(
            [
                v["regime"] if j == 0 else "",
                emph(allocation_label(model)) if model == "SP500" else allocation_label(model),
                *vals,
            ]
        )
        short_label = allocation_label(model).replace("Constant-corr. GMVP", "CC GMVP")
        block += row([emph(short_label) if model == "SP500" else short_label, *vals])
    sub += "\\addlinespace[2pt]\n"
    regime_blocks.append(block + "\\addlinespace[4pt]\n")
replace_rows("tab:coverage", coverage_table)
a = text.index(r"\label{tab:period}")
b = text.index(r"\end{table*}", a)
period_table = text[a:b]
if period_table.count(r"\begin{minipage}") == 2:
    split = (len(regime_blocks) + 1) // 2
    panels = iter(["".join(regime_blocks[:split]), "".join(regime_blocks[split:])])
    period_table, count = re.subn(
        r"\\midrule\s*.*?\\bottomrule",
        lambda _: r"\midrule" + "\n" + next(panels) + r"\bottomrule",
        period_table,
        flags=re.S,
    )
    assert count == 2, "Expected two six-column regime panels"
    text = text[:a] + period_table + text[b:]
else:
    replace_rows("tab:period", sub)

mechanism_table = ""
for model in names:
    v = D.loc[model]
    vals = [
        f"{v.effective_n:.1f}",
        f"{v.cap_count:.1f}",
        pct(v.tilt_score1),
        pct(v.tilt_score2),
        pct(v.tilt_score3),
        pct(v.delta_annual_gross_mean),
        pct(v.delta_annual_cost_drag),
        pct(v.delta_annual_net_mean),
    ]
    if model == "gmvp_constant_corr":
        vals[3] = emph(vals[3])
        vals[4] = emph(vals[4], "underline")
    if model in ["gmvp_sample", "tc_gmvp"]:
        vals[6] = emph(vals[6])
        vals[7] = emph(vals[7])
    mechanism_table += row([allocation_label(model), *vals])
replace_rows("tab:mechanism", mechanism_table)

costs = ""
for c in [0, 10, 25, 50, 75, 100]:
    vals = [
        get(model, f"total_observed_drift{c}").cagr
        for model in ["source_weight", "equal_weight", "gmvp_constant_corr", "tc_gmvp"]
    ]
    costs += row(
        [
            emph(str(c), "underline") if c == 10 else c,
            *[emph(pct(v)) if v == max(vals) else pct(v) for v in vals],
        ]
    )
replace_rows("tab:cost", costs)
# Reviewed narrative is maintained in the article; rebuilding does not overwrite it.

# Refresh the figures from the saved return series when rebuilding the article.
from .figures import build_figures, embed_figures

text = embed_figures(text, build_figures())

# Keep the selection-risk check in the actual report when rebuilding it.
selection_section = P / "selection_comparison/article_section.tex"
if selection_section.exists():
    begin = "% BEGIN SELECTION RISK AUDIT"
    end = "% END SELECTION RISK AUDIT"
    if begin in text:
        a = text.index(begin)
        b = text.index(end, a) + len(end)
        text = text[:a] + text[b:].lstrip("\n")
    anchor = r"\subsubsection{Interpretation and Remaining Limits}"
    text = text.replace(anchor, selection_section.read_text() + anchor, 1)

# Generate actual IEEEtran BibTeX output, then embed it to retain a standalone
# source compatible with the native editor's single-file compiler.
boundary = "% Bibliography generated from references.bib with IEEEtran.bst."
body = text[: text.index(boundary) if boundary in text else text.index(r"\begin{thebibliography}")]
order = list(
    dict.fromkeys(
        k.strip()
        for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", body)
        for k in group.split(",")
    )
)
bib = PAPER / "references.bib"
with TemporaryDirectory(prefix="ife_ieee_bib_") as directory:
    q = Path(directory)
    (q / "references.bib").write_text(bib.read_text())
    (q / "references.aux").write_text(
        "\\relax\n"
        + "".join("\\citation{" + key + "}\n" for key in order)
        + "\\bibstyle{IEEEtran}\n\\bibdata{references}\n"
    )
    result = subprocess_run(["bibtex", "references"], cwd=q, capture_output=True, text=True)
    log = (q / "references.blg").read_text()
    if result.returncode or "Warning--" in log:
        raise RuntimeError(result.stdout + "\n" + log)
    bbl = (q / "references.bbl").read_text()
    assert set(re.findall(r"\\bibitem\{([^}]+)\}", bbl)) == set(order)
    text = (
        body
        + "% Bibliography generated from references.bib with IEEEtran.bst.\n"
        + bbl
        + "\n"
        + r"\end{document}"
        + "\n"
    )
    (article.parent / "references.bbl").write_text(bbl)
article.write_text(text)
print(
    f"Generated report: {len(full)} models, {len(coverage)} observed regimes, {len(order)} IEEE references."
)
