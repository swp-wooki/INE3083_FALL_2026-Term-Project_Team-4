"""Compare frozen selections using the article's observed-return accounting.

Run from the project root, or call run() from a notebook. This evaluates equal
weights and the supplied market-capitalization weights, not new optimizers.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import experiment
from .config import Config
from .metrics import hac_regression, performance
from .models import module
from .paths import DATA, OUTPUT, ROOT
from .provenance import file_hash
from .selection import holm_adjust, load_selection

ARTICLE = OUTPUT
OUTPUT = OUTPUT / "selection_comparison"


def write_article_section(metrics, tests, output=OUTPUT):
    """Emit the standalone section consumed by the existing article builder."""
    full = metrics.query("period == 'full'").set_index(["selection", "model"])
    later = metrics.query("period == '2012_2020'").set_index(["selection", "model"])
    paired = tests.set_index("model")
    rows = []
    for selection, label in [
        ("defensive", "Defensive (30)"),
        ("return", "Return-oriented (50)"),
    ]:
        for model, name in [
            ("equal_weight", "Equal weight"),
            ("source_weight", "Market-cap weight"),
        ]:
            r = full.loc[selection, model]
            rows.append(
                f"{label} & {name} & {100 * r.cagr:.2f} & {100 * r.annual_volatility:.2f} & {r.sharpe:.3f} & {100 * r.max_drawdown:.2f}"
                + r" \\"
            )
    old, new = [full.loc[tag, "source_weight"] for tag in ["defensive", "return"]]
    test = paired.loc["source_weight"]
    section = (
        r"""% BEGIN SELECTION RISK AUDIT
\subsubsection{Return-Oriented versus Defensive Selection}\label{sec:selectionrisk}
We separately compare the supplied momentum--profitability--issuance selection (50 stocks) with the low-risk--coskewness defensive selection (30 stocks). Both use February 2003--December 2020, observed CRSP total returns, drift-adjusted transaction costs of 10 basis points per dollar traded, and terminal proceeds held in cash. Constituents are retained exactly; no new size filter or missing-return imputation is applied. The two allocation benchmarks are equal weighting and recorded market-capitalization weights, with the common rule $u=2/N$ giving limits of 4.00\% and 6.67\%, respectively. Reconstructed proportional weights agree with both supplied files within $5\times10^{-7}$, and the defensive monthly net returns reproduce the article baseline within $10^{-12}$.

\begin{table*}[!htbp]\centering
\caption{Selection comparison under the article's return and cost conventions. CAGR, annualized volatility, and maximum drawdown (MDD) are percentages; Sharpe is dimensionless. Selection and portfolio size change together. These are allocation benchmarks, not a re-estimation of all optimizers on the new universe.}\label{tab:selectionrisk}
\small\begin{tabular}{llrrrr}\toprule
Selection ($N$) & Allocation & CAGR & Volatility & Sharpe & MDD\\\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule\end{tabular}\end{table*}

"""
    )
    section += (
        f"Under market-capitalization weighting, CAGR rises by {100 * (new.cagr - old.cagr):.2f} percentage points, "
        f"but volatility rises by {100 * (new.annual_volatility - old.annual_volatility):.2f} points and Sharpe falls "
        f"from {old.sharpe:.3f} to {new.sharpe:.3f}. Maximum drawdown becomes {100 * (old.max_drawdown - new.max_drawdown):.2f} "
        "percentage points more severe. Equal weighting gives the same qualitative result. Thus the observed growth gain comes with materially greater realized risk; the new selection does not establish a return improvement at comparable risk.\n\n"
    )
    section += (
        "For market-capitalization weights, a paired six-month circular block bootstrap (2,000 draws) gives pointwise 95\\% intervals of "
        f"$[{100 * test.cagr_difference_ci_low:.2f},{100 * test.cagr_difference_ci_high:.2f}]$ percentage points for the CAGR difference, "
        f"$[{100 * test.volatility_difference_ci_low:.2f},{100 * test.volatility_difference_ci_high:.2f}]$ points for the volatility difference, "
        f"and $[{test.sharpe_difference_ci_low:.3f},{test.sharpe_difference_ci_high:.3f}]$ for the Sharpe difference. "
        "All differences are return-oriented minus defensive. The volatility increase is clear, whereas the CAGR interval includes zero. "
        f"The paired mean-return HAC(4) test has $p={test.hac_p:.3f}$, or $p={test.holm_p:.3f}$ after Holm adjustment "
        "across the two weighting rules. A Sharpe interval containing zero is not an equivalence test or evidence that risk is unchanged.\n\n"
    )
    de, re = [later.loc[tag, "equal_weight"] for tag in ["defensive", "return"]]
    dm, rm = [later.loc[tag, "source_weight"] for tag in ["defensive", "return"]]
    section += (
        f"In 2012--2020, equal-weight CAGR is {100 * re.cagr:.2f}\\% for the return-oriented selection versus {100 * de.cagr:.2f}\\% "
        f"for the defensive selection. Market-capitalization weighting retains a CAGR increase ({100 * rm.cagr:.2f}\\% versus {100 * dm.cagr:.2f}\\%), "
        f"but volatility remains higher ({100 * rm.annual_volatility:.2f}\\% versus {100 * dm.annual_volatility:.2f}\\%) and Sharpe lower "
        f"({rm.sharpe:.3f} versus {dm.sharpe:.3f}). The comparison is conditional on historical selections: different stock counts, "
        "associated position limits, and prior factor searches prevent attributing the difference solely to the factors. "
        "Optimization may still improve risk-adjusted performance within the new selection, but that is a separate comparison against its own equal-weight baseline.\n"
        "% END SELECTION RISK AUDIT\n\n"
    )
    (Path(output) / "article_section.tex").write_text(section)


def run(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "defensive": DATA / "selections/defensive_selection.csv",
        "return": DATA / "selections/return_oriented_selection.csv",
    }
    selections = {k: load_selection(p, n) for (k, p), n in zip(paths.items(), (30, 50))}
    months = pd.PeriodIndex(sorted(selections["defensive"].holding_month.unique()), freq="M")
    assert months.equals(
        pd.PeriodIndex(sorted(selections["return"].holding_month.unique()), freq="M")
    )
    ids = set().union(*(set(s.PERMNO) for s in selections.values()))
    raw_path = Config().data_dir / "raw/security_monthly.csv.gz"
    chunks = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=["PERMNO", "YYYYMM", "MthRet", "MthCap", "SecurityEndDt"],
        chunksize=250_000,
    ):
        chunks.append(chunk.loc[chunk.PERMNO.isin(ids)])
    raw = pd.concat(chunks, ignore_index=True).drop_duplicates()
    raw["month"] = pd.PeriodIndex(raw.YYYYMM.astype(str), freq="M")
    assert not raw.duplicated(["PERMNO", "month"]).any()
    ends = raw[["PERMNO", "SecurityEndDt"]].dropna().drop_duplicates()
    assert not ends.PERMNO.duplicated().any()
    end_month = pd.to_datetime(ends.set_index("PERMNO").SecurityEndDt).dt.to_period("M")
    total = raw.pivot(index="month", columns="PERMNO", values="MthRet").apply(pd.to_numeric)
    size = raw.pivot(index="month", columns="PERMNO", values="MthCap").apply(pd.to_numeric)
    rf = pd.read_csv(ARTICLE / "benchmark_returns.csv", index_col="month").RF
    rf.index = pd.PeriodIndex(rf.index, freq="M")
    rows, checks = [], {}
    for tag, selected in selections.items():
        names = pd.Index(sorted(selected.PERMNO.unique()))
        n = int(selected.groupby("holding_month").size().iloc[0])
        cap = 2 / n
        realized = pd.DataFrame(0.0, index=months, columns=names)
        ended = pd.DataFrame(
            months.values[:, None] == end_month.reindex(names).values[None, :],
            index=months,
            columns=names,
        )
        targets = {m: realized.copy() for m in ["equal_weight", "source_weight"]}
        reconstruction_errors, missing_sizes = [], 0
        for month, group in selected.groupby("holding_month"):
            held = pd.Index(group.PERMNO)
            ret = total.loc[month, held]
            if not np.isfinite(ret).all() or ret.lt(-1).any():
                raise ValueError(f"Invalid observed return: {tag}, {month}")
            realized.loc[month, held] = ret
            recorded = group.set_index("PERMNO").weight.reindex(held).to_numpy()
            assert np.isfinite(recorded).all() and (recorded >= 0).all()
            assert abs(recorded.sum() - 1) <= 2e-5 and recorded.max() <= cap + 1e-6
            targets["equal_weight"].loc[month, held] = 1 / n
            targets["source_weight"].loc[month, held] = module("factor_cov")._proj(recorded, cap)
            caps = size.loc[month - 1, held]
            if np.isfinite(caps).all() and caps.gt(0).all():
                reconstruction_errors.append(
                    float(np.max(np.abs(experiment.capped_proportional(caps, cap) - recorded)))
                )
            else:
                missing_sizes += 1
        checks[tag] = dict(
            n=n,
            cap=cap,
            held_observations=len(selected),
            missing_held_returns=0,
            months_missing_formation_size=missing_sizes,
            max_recorded_vs_proportional_weight_error=max(reconstruction_errors, default=None),
        )
        if reconstruction_errors and max(reconstruction_errors) > 2e-6:
            raise ValueError(f"Recorded weights are inconsistent with market-cap weighting: {tag}")
        for model, weights in targets.items():
            assert np.allclose(weights.sum(axis=1), 1) and weights.max().max() <= cap + 1e-10
            rows.append(
                experiment.evaluate(weights, realized, ended, 10).assign(selection=tag, model=model)
            )
    returns = pd.concat(rows, ignore_index=True)
    old = pd.read_csv(ARTICLE / "monthly_returns.csv").query(
        "scenario == 'total_observed_drift10' and model in ['equal_weight', 'source_weight']"
    )
    aligned = returns.query("selection == 'defensive'").merge(
        old,
        on=["month", "model"],
        validate="one_to_one",
        suffixes=("_audit", "_article"),
    )
    assert len(aligned) == 2 * len(months)
    np.testing.assert_allclose(
        aligned.net_return_audit, aligned.net_return_article, atol=1e-12, rtol=0
    )
    returns.to_csv(output / "monthly_returns.csv", index=False)
    periods = {
        "full": (str(months[0]), str(months[-1])),
        "2003_2011": ("2003-02", "2011-12"),
        "2012_2020": ("2012-01", "2020-12"),
    }
    stats = []
    for (selection, model), group in returns.groupby(["selection", "model"]):
        group = group.set_index("month")
        group.index = pd.PeriodIndex(group.index, freq="M")
        for period, (start, end) in periods.items():
            sample = group.loc[start:end]
            stats.append(
                dict(
                    selection=selection,
                    model=model,
                    period=period,
                    **performance(sample.net_return, rf),
                    mean_turnover=sample.turnover.mean(),
                )
            )
    metrics = pd.DataFrame(stats)
    metrics.to_csv(output / "metrics.csv", index=False)
    wide = returns.pivot(
        index="month", columns=["selection", "model"], values="net_return"
    ).sort_index()
    rng = np.random.default_rng(20261009)
    t = len(wide)
    starts = rng.integers(t, size=(2000, (t + 5) // 6))
    index = ((starts[:, :, None] + np.arange(6)) % t).reshape(2000, -1)[:, :t]
    rf_values = rf.reindex(pd.PeriodIndex(wide.index, freq="M")).to_numpy()
    inference = []
    for model in ["equal_weight", "source_weight"]:
        new, old = [wide[tag, model].to_numpy() for tag in ["return", "defensive"]]
        beta, stat, p = hac_regression(new - old, lags=4)
        boot = []
        for values in [new, old]:
            sample = values[index]
            excess = sample - rf_values[index]
            boot.append(
                np.column_stack(
                    [
                        np.expm1(np.log1p(sample).mean(axis=1) * 12),
                        sample.std(axis=1, ddof=1) * np.sqrt(12),
                        excess.mean(axis=1) / excess.std(axis=1, ddof=1) * np.sqrt(12),
                    ]
                )
            )
        ci = np.quantile(boot[0] - boot[1], [0.025, 0.975], axis=0)
        result = dict(model=model, annual_mean_difference=12 * beta[0], hac_t=stat[0], hac_p=p[0])
        for j, field in enumerate(["cagr", "volatility", "sharpe"]):
            result.update(
                {
                    f"{field}_difference_ci_low": ci[0, j],
                    f"{field}_difference_ci_high": ci[1, j],
                }
            )
        inference.append(result)
    tests = pd.DataFrame(inference)
    tests["holm_p"] = holm_adjust(tests.hac_p)
    tests.to_csv(output / "paired_inference.csv", index=False)
    write_article_section(metrics, tests, output)
    manifest = dict(
        accounting="Observed raw CIZ MthRet; drift-adjusted one-way 10bp; terminal residual to cash; no return imputation or clipping",
        period=periods["full"],
        bootstrap=dict(replications=2000, block_months=6, seed=20261009),
        selections={
            k: dict(path=str(p.relative_to(ROOT)), sha256=file_hash(p), **checks[k])
            for k, p in paths.items()
        },
        article_return_max_error=float(
            (aligned.net_return_audit - aligned.net_return_article).abs().max()
        ),
        scope="Two allocation benchmarks only; N=50 vs N=30 confounds a pure factor effect; no optimizer re-estimation",
        source_return_path=str(raw_path.relative_to(ROOT)),
        source_return_bytes=raw_path.stat().st_size,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for (selection, model), series in wide.items():
        wealth = (1 + series).cumprod()
        dates = pd.PeriodIndex(series.index, freq="M").to_timestamp("M")
        label = f"{selection.title()} N{checks[selection]['n']}: " + (
            "Equal weight" if model == "equal_weight" else "Market-cap weight"
        )
        axes[0].plot(dates, wealth - 1, label=label)
        axes[1].plot(dates, wealth / wealth.cummax().clip(lower=1) - 1)
    for ax, title in zip(axes, ["Cumulative net return", "Drawdown"]):
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.2)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "selection_risk_comparison.png", dpi=160)
    plt.close(fig)
    print(
        metrics.query("period == 'full'")[
            [
                "selection",
                "model",
                "cagr",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(tests.round(4).to_string(index=False))
    print(json.dumps(checks))
    return metrics, tests


if __name__ == "__main__":
    run()
