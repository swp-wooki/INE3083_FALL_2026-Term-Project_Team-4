"""Rebuild the reviewed article's numerical tables and figures from this run."""

import json
import re
from pathlib import Path
from subprocess import run as subprocess_run
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .figures import PRIMARY_MARKET, build_figures, embed_figures
from .metrics import performance
from .paths import OUTPUT, PAPER, ROOT


def table_bounds(text, label):
    """Find this labeled table; never match a rule or end marker in the next float."""
    tag = r"\label{" + label + "}"
    if text.count(tag) != 1:
        raise ValueError(f"Expected exactly one label {label!r}")
    start = text.index(tag)
    boundary = re.search(r"\\(begin|end)\{table\*?\}", text[start:])
    if boundary is None or boundary.group(1) != "end":
        raise ValueError(f"No table end found for {label}")
    return start, start + boundary.start()


def replace_rows(text, label, rows, panel=None):
    """Replace data rows only, preserving neighboring panels, notes, and narrative."""
    start, stop = table_bounds(text, label)
    table = text[start:stop]
    anchor = 0
    if panel:
        marker = r"\textit{Panel " + panel + "."
        if table.count(marker) != 1:
            raise ValueError(f"Missing or duplicate panel {panel} in {label}")
        anchor = table.index(marker)
    header = table.index(r"\midrule", anchor) + len(r"\midrule")
    end = table.index(r"\bottomrule", header)
    # Some panels share a single tabular; their boundary is a midrule, not a bottomrule.
    next_panel = table.find(r"\textit{Panel ", header, end)
    if next_panel != -1:
        end = table.rindex(r"\midrule", header, next_panel)
    if not rows.strip() or re.search(
        r"\\(?:begin|end)\{(?:table\*?|tabular)\}|\\(?:bottomrule|section|label)\b", rows
    ):
        raise ValueError(f"Expected data rows only for {label}")
    return text[: start + header] + "\n" + rows.rstrip() + "\n" + text[start + end :]


def validate_selection_section(text):
    """Reject a malformed reviewed section before any generated content is inserted."""
    markers = ("% BEGIN SELECTION RISK AUDIT", "% END SELECTION RISK AUDIT")
    matches = [
        list(re.finditer(r"^" + re.escape(marker) + r"[ \t]*$", text, re.M)) for marker in markers
    ]
    if any(len(group) != 1 for group in matches):
        raise ValueError(
            "Expected one matched BEGIN/END SELECTION RISK AUDIT pair in the reviewed manuscript"
        )
    start, end = (group[0].start() for group in matches)
    body = text[start:end]
    if body.count(r"\label{tab:selectionrisk}") != 1 or re.search(
        r"\\(?:section\{|appendices\b)", body
    ):
        raise ValueError(
            "Selection-risk markers must enclose Table IX without consuming the following section"
        )


def emph(value, kind="textbf"):
    return "\\" + kind + "{" + value + "}"


def main():
    from .allocation_sensitivity import render_calibration
    from .revised_inference import render_market_panel
    from .selection_revisions import render_prose

    P = OUTPUT
    text = (PAPER / "main.tex").read_text()
    validate_selection_section(text)
    article = P / "paper/main.tex"
    article.parent.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(P / "model_metrics.csv")
    R = pd.read_csv(P / "monthly_returns.csv")
    Q = pd.read_csv(P / "paired_comparisons.csv")
    B = pd.read_csv(P / "benchmark_returns.csv", index_col=0)
    BM = pd.read_csv(P / "benchmark_metrics.csv")
    A = pd.read_csv(P / "data_audit.csv")
    S = pd.read_csv(P / "frozen_selection.csv")
    PRIMARY = "total_observed_drift10"
    MARKETS = [PRIMARY_MARKET, "FF_market", "SP500"]
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
        "source_weight": "Capped VW",
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

    assert np.isfinite(
        full[["cagr", "sharpe", "annual_volatility", "max_drawdown"]].to_numpy()
    ).all()
    # Paired circular block bootstrap: identical resampled months across all models and RF.
    wide = R.query("scenario == @PRIMARY").pivot(
        index="month", columns="model", values="net_return"
    )
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
        assert np.isclose(
            (1 + g.cumulative_return).prod() - 1, expected.cumulative_return, atol=1e-12
        )
    assert np.isfinite(G[["cagr", "annual_volatility", "sharpe", "max_drawdown"]].to_numpy()).all()
    C.to_csv(P / "regime_coverage.csv", index=False)
    G.to_csv(P / "regime_metrics.csv", index=False)

    D = pd.read_csv(P / "mechanism_summary.csv").set_index("model")
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
    main += r"\midrule" + "\n"
    for model in MARKETS:
        v = BM.query('model == @model and period == "full"').iloc[0]
        vals = [
            pct(v.cagr),
            pct(v.annual_volatility),
            f"{v.sharpe:.3f}",
            pct(v.max_drawdown),
        ]
        if model == PRIMARY_MARKET:
            vals = [emph(x, "underline") for x in vals]
        main += row(
            [
                emph(allocation_label(model))
                if model == PRIMARY_MARKET
                else allocation_label(model),
                *vals,
                "--",
            ]
        )
    text = replace_rows(text, "tab:main", main)

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
    text = replace_rows(text, "tab:inference", inference, panel="A")
    text = render_market_panel(text, P)

    coverage_table = ""
    regime_blocks = []
    for v in coverage:
        observed = v["observed_start"] + "--" + v["observed_end"]
        count = str(v["months"])
        if v["regime"] in ["Pre-Crisis", "Crisis II"]:
            observed = emph(observed)
        if v["months"] < 12:
            count = (
                emph(count, "underline") if v["regime"] in ["Crisis II", "Recovery II"] else count
            )
        coverage_table += row([v["regime"], v["type"], observed, count])
        subset = G[
            (G.regime == v["regime"])
            & G.model.isin(DISPLAY)
            & G.scenario.isin([PRIMARY, "benchmark"])
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
            if model == PRIMARY_MARKET:
                vals[0] = emph(vals[0], "underline")
            short_label = allocation_label(model).replace("Constant-corr. GMVP", "CC GMVP")
            block += row([emph(short_label) if model == PRIMARY_MARKET else short_label, *vals])
        regime_blocks.append(block + "\\addlinespace[4pt]\n")
    text = replace_rows(text, "tab:coverage", coverage_table)
    a, b = table_bounds(text, "tab:period")
    period_table = text[a:b]
    split = (len(regime_blocks) + 1) // 2
    panels = ["".join(regime_blocks[:split]), "".join(regime_blocks[split:])]
    matches = list(re.finditer(r"\\midrule\s*.*?\\bottomrule", period_table, flags=re.S))
    if len(matches) != len(panels):
        raise ValueError("Expected two six-column regime panels in tab:period")
    for match, panel in reversed(list(zip(matches, panels))):
        period_table = (
            period_table[: match.start()]
            + r"\midrule"
            + "\n"
            + panel
            + r"\bottomrule"
            + period_table[match.end() :]
        )
    text = text[:a] + period_table + text[b:]

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
    text = replace_rows(text, "tab:mechanism", mechanism_table)

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
    text = replace_rows(text, "tab:cost", costs)
    # Table IX is the reviewed four-signal/blend experiment, not the older two-selection section.
    revision_dir = P / "revisions"
    for label, panel, fragment in [
        ("tab:archive", "A", "archive_a.tex"),
        ("tab:archive", "B", "archive_b.tex"),
        ("tab:selectionrisk", None, "selectionrisk.tex"),
        ("tab:extwindows", None, "extwindows.tex"),
        ("tab:extconstraints", None, "extconstraints.tex"),
    ]:
        text = replace_rows(text, label, (revision_dir / fragment).read_text(), panel)
    text = render_calibration(render_prose(text, revision_dir), revision_dir)
    validate_selection_section(text)
    text = embed_figures(text, build_figures())

    # Generate actual IEEEtran BibTeX output, then embed it to retain a standalone
    # source compatible with the native editor's single-file compiler.
    boundary = "% Bibliography generated from references.bib with IEEEtran.bst."
    body = text[
        : text.index(boundary) if boundary in text else text.index(r"\begin{thebibliography}")
    ]
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


if __name__ == "__main__":
    main()
