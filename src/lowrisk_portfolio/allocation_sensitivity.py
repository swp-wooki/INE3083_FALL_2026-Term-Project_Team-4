"""Reproduce Appendix C from observed returns, formation scores, sectors and BAB.

The catalogue distinguishes the 49 additional settings from five reused allocations,
the capped value-weighted comparator, and the separate 36-month EWMA sensitivity.
"""

import json
import re
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_panel
from .experiment import CAP, PRIMARY_COST_BPS, evaluate
from .metrics import hac_regression, performance
from .models import module, validate_weights
from .paths import DATA, OUTPUT, PAPER, ROOT
from .selection import holm_adjust, load_selection

KINDS = ("sample", "cc", "lw")
ESTIMATOR = {"sample": "cov_sample", "cc": "cov_constant_corr", "lw": "cov_lw_identity"}
CAPS = {"2K": CAP, "3K": 3 / 30, "none": 1.0}
FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
BASELINES = {
    "source_weight": "source_weight",
    "sample_36_2K": "gmvp_sample",
    "cc_36_2K": "gmvp_constant_corr",
    "lw_36_2K": "gmvp_lw_identity",
    "ff5_36_2K": "gmvp_factor",
    "sample_erc_cap": "erc",
}


@dataclass(frozen=True)
class Setting:
    name: str
    kind: str
    window: int
    cap: float
    rule: str
    family: str

    @property
    def role(self):
        if self.name in BASELINES:
            return "baseline"
        return "diagnostic" if self.name == "ewma_36" else "additional"


def catalogue():
    specs = [Setting("source_weight", "sample", 36, CAP, "source", "baseline")]
    for kind, windows in (
        ("sample", (36, 60)),
        ("lw", (24, 36, 60)),
        ("ff5", (36, 60)),
        ("ff5bab", (36, 60)),
    ):
        cap_labels = ("2K", "none") if kind.startswith("ff5") else tuple(CAPS)
        for window in windows:
            for label in cap_labels:
                specs.append(
                    Setting(
                        f"{kind}_{window}_{label}",
                        kind,
                        window,
                        CAPS[label],
                        "gmvp",
                        "factor_covariance" if kind.startswith("ff5") else "window_cap",
                    )
                )
    specs.append(Setting("cc_36_2K", "cc", 36, CAP, "gmvp", "baseline"))
    for kind in KINDS:
        for rule, family in (
            ("match_bvc", "match_bvc"),
            ("match_c", "match_c"),
            ("te2", "tracking_error"),
            ("te3", "tracking_error"),
            ("neff20", "diversification"),
            ("sic25", "diversification"),
            ("sic30", "diversification"),
            ("erc_cap", "risk_budgeting"),
            ("erc_nocap", "risk_budgeting"),
        ):
            specs.append(
                Setting(
                    f"{kind}_{rule}", kind, 36, 1.0 if rule == "erc_nocap" else CAP, rule, family
                )
            )
    for w in (6, 12):
        specs.append(Setting(f"lw_short_{w}", "lw", w, CAP, "gmvp", "short_window"))
    specs.extend(
        [
            Setting("ewma_expanding", "ewma", 0, CAP, "gmvp", "short_window"),
            Setting("ewma_36", "ewma", 36, CAP, "gmvp", "short_window"),
        ]
    )
    if (
        len({s.name for s in specs}) != len(specs)
        or sum(s.role == "additional" for s in specs) != 49
    ):
        raise ValueError("Invalid Appendix C experiment catalogue")
    return specs


def _solvers():
    spec = spec_from_file_location("extension_solvers", ROOT / "extensions/solvers.py")
    solvers = module_from_spec(spec)
    spec.loader.exec_module(solvers)
    return solvers


def _raw_industries(selection, clean):
    """Restore missing formation-month SICs from the security record valid that month."""
    required = pd.MultiIndex.from_frame(selection[["formation_month", "PERMNO"]])
    row = clean.index.get_indexer(required.get_level_values(0))
    col = clean.columns.get_indexer(required.get_level_values(1))
    if (row < 0).any() or (col < 0).any():
        raise ValueError("Formation keys outside the course panel")
    needed = required[pd.isna(clean.to_numpy()[row, col])]
    if not len(needed):
        return pd.DataFrame()
    wanted = set(
        zip(
            [int(str(m).replace("-", "")) for m in needed.get_level_values(0)],
            needed.get_level_values(1),
        )
    )
    ids = needed.get_level_values(1).unique()
    chunks = []
    use = ["PERMNO", "YYYYMM", "SICCD", "SecInfoStartDt", "SecInfoEndDt", "MthCalDt"]
    for part in pd.read_csv(DATA / "raw/security_monthly.csv.gz", usecols=use, chunksize=250_000):
        part = part.loc[part.PERMNO.isin(ids)]
        hit = pd.MultiIndex.from_frame(part[["YYYYMM", "PERMNO"]]).isin(wanted)
        chunks.append(part.loc[hit])
    source = pd.concat(chunks).dropna(subset=["SICCD"])
    duplicated = source.groupby(["YYYYMM", "PERMNO"]).SICCD.transform("nunique").gt(1)
    active = source.SecInfoStartDt.le(source.MthCalDt) & (
        source.SecInfoEndDt.isna() | source.SecInfoEndDt.ge(source.MthCalDt)
    )
    source = source.loc[~duplicated | active, ["YYYYMM", "PERMNO", "SICCD"]].drop_duplicates()
    source["month"] = pd.PeriodIndex(source.YYYYMM.astype(str), freq="M")
    return source.pivot(index="month", columns="PERMNO", values="SICCD")


def _covariance(kind, hist, factors):
    x = hist.to_numpy()
    if kind in ESTIMATOR:
        return module(ESTIMATOR[kind]).estimate(x)
    if kind == "ewma":
        weights = 0.94 ** np.arange(len(x) - 1, -1, -1)
        weights /= weights.sum()
        centered = x - weights @ x
        return centered.T @ (weights[:, None] * centered)
    cols = FF5 + (["BAB"] if kind == "ff5bab" else [])
    aligned = factors.loc[hist.index, [*cols, "RF"]]
    if aligned.isna().any().any():
        raise ValueError("Incomplete aligned factor history")
    return module("factor_cov").factor_cov(hist.sub(aligned.RF, axis=0), aligned[cols]).to_numpy()


def _gmvp(S, cap):
    """Certify the capped-simplex solution by its linearization gap; refine if needed."""
    solver = module("factor_cov").gmvp

    def gap(w):
        g = 2 * S @ w
        budget = np.clip(1 - cap * np.arange(len(w)), 0, cap)
        return float(max(g @ w - np.sort(g) @ budget, 0) / (w @ S @ w))

    w = solver(S, cap=cap)
    if gap(w) > 1e-7:
        w = solver(S, cap=cap, w0=w, iters=400_000, tol=1e-13)
    bound = gap(w)
    if bound > 1e-6:
        raise ValueError(f"Uncertified GMVP (relative gap {bound:.3g})")
    return w, {"relative_gap": bound}


def _allocate(setting, S, vw, scores, sector, solvers):
    rule, cap = setting.rule, setting.cap
    if rule == "gmvp":
        return _gmvp(S, cap)
    if rule == "source":
        return vw.copy(), {}
    if rule.startswith("match_"):
        A = np.column_stack((np.ones(len(vw)), scores if rule == "match_bvc" else scores[:, 2])).T
        return solvers.exposure_neutral_gmvp(S, A, A @ vw, cap, vw)
    if rule.startswith("te"):
        return solvers.te_limited_gmvp(S, vw, int(rule[2:]) / 100, cap)
    if rule.startswith("sic"):
        return solvers.sector_capped_gmvp(S, sector, int(rule[3:]) / 100, cap)
    if rule == "neff20":
        return solvers.effective_n_gmvp(S, 20, cap)
    if rule in ("erc_cap", "erc_nocap"):
        return module("erc").constrained_risk_budgeting(S, cap, return_info=True)
    raise ValueError(f"Unknown allocation rule {rule}")


def _table_rows(metrics):
    """Format the two result grids in their reviewed, chronological table order."""
    index = metrics.loc[metrics.period.eq("full")].set_index("variant").sharpe

    def fmt(key, bold=False):
        value = f"{index.loc[key]:.3f}"
        return r"\textbf{" + value + "}" if bold else value

    window_rows = []
    for kind, label, windows in [
        ("sample", "Sample", [36, 60]),
        ("lw", "LW identity", [24, 36, 60]),
        ("ff5", "FF5", [36]),
        ("ff5bab", "FF5 + BAB", [36]),
        ("ff5", "FF5", [60]),
        ("ff5bab", "FF5 + BAB", [60]),
    ]:
        for win in windows:
            cells = [label, str(win) if win != 60 else r"$\leq60$"]
            for cap in CAPS:
                key = f"{kind}_{win}_{cap}"
                cells.append(
                    fmt(key, kind in ("sample", "lw") and win == 36 and cap == "none")
                    if key in index
                    else "--"
                )
            window_rows.append(
                " & ".join(cells) + r"\\" + (r"\midrule" if (kind, win) == ("lw", 60) else "")
            )
    constraint_rows = []
    for rule, label in [
        ("36_2K", "Reference GMVP"),
        ("match_bvc", "Match $B,V,C$ scores"),
        ("match_c", "Match $C$ score only"),
        ("te2", r"TE $\leq2\%$"),
        ("te3", r"TE $\leq3\%$"),
        ("neff20", r"Effective holdings $\geq20$"),
        ("sic25", r"Sector weight $\leq25\%$"),
        ("sic30", r"Sector weight $\leq30\%$"),
        ("erc_cap", "Risk budgeting, $2/K$ cap"),
        ("erc_nocap", "Equal risk contributions, no cap"),
    ]:
        cells = [
            label,
            *[
                fmt(f"{kind}_{rule}", kind == "cc" and rule in ("36_2K", "match_bvc", "sic25"))
                for kind in KINDS
            ],
        ]
        constraint_rows.append(" & ".join(cells) + r"\\" + (r"\midrule" if rule == "sic30" else ""))
    return {
        "extwindows": "\n".join(window_rows) + "\n",
        "extconstraints": "\n".join(constraint_rows) + "\n",
    }


def verify_table_rows(manuscript, label, rows):
    """Compare each recomputed row with the submitted table at its printed precision."""
    if manuscript.count(r"\label{tab:" + label + "}") != 1:
        raise ValueError(f"Expected one labeled table tab:{label}")
    block = manuscript.split(r"\label{tab:" + label + "}", 1)[1]
    block = block.split(r"\bottomrule", 1)[0].split(r"\midrule", 1)[1]
    norm = lambda line: re.sub(r"\s+", "", line)
    expected = [norm(line) for line in block.splitlines() if "&" in line]
    actual = [norm(line) for line in rows.splitlines() if "&" in line]
    if len(expected) != len(actual):
        raise ValueError(f"Row count mismatch in tab:{label}: {len(actual)} vs {len(expected)}")
    for i, (found, wanted) in enumerate(zip(actual, expected), start=1):
        if found != wanted:
            raise ValueError(f"tab:{label} row {i}: computed {found}; manuscript {wanted}")
    return len(actual)


def _risk_ranges(risk, specs):
    """Compare net realized volatility with mean month-ahead volatility, on named populations."""
    answer = {}
    for label, valid in [
        ("fixed_36_month_gmvp", lambda s: s.window == 36),
        ("36_or_longer_window_gmvp", lambda s: s.window >= 36),
    ]:
        selected = [
            s.name
            for s in specs
            if valid(s)
            and s.role != "diagnostic"
            and s.rule not in ("source", "erc_cap", "erc_nocap")
        ]
        ratios = risk.loc[selected, "realized_over_forecast"]
        answer[label] = dict(
            settings=selected,
            count=len(selected),
            minimum=float(ratios.min()),
            minimum_setting=ratios.idxmin(),
            maximum=float(ratios.max()),
            maximum_setting=ratios.idxmax(),
        )
    return answer


def render_calibration(text, revision_dir):
    """Refresh the two risk ranges in the reviewed text without replacing the discussion."""
    summary = json.loads((Path(revision_dir) / "allocation_summary.json").read_text())
    fixed = summary["risk_calibration"]["fixed_36_month_gmvp"]
    longer = summary["risk_calibration"]["36_or_longer_window_gmvp"]
    lo2, hi2 = summary["realized_te_limit_2pct"]
    lo3, hi3 = summary["realized_te_limit_3pct"]
    parts = [
        (
            r"Across the [^\n]*?fixed 36-month GMVP specifications,[^\n]*?(?= The numerator is the annualized standard deviation of net monthly returns;)",
            f"Across the {fixed['count']} fixed 36-month GMVP specifications, realized annualized volatility was "
            f"{fixed['minimum']:.2f}--{fixed['maximum']:.2f} times its mean ex ante forecast. "
            f"Including the longer windows gives a range of {longer['minimum']:.2f}--{longer['maximum']:.2f} "
            f"across all {longer['count']} GMVP specifications.",
        ),
        (
            r"Realized tracking error ranges from [^\n]*?(?= Moving closer to Capped VW)",
            rf"Realized tracking error ranges from {100 * lo2:.2f}\% to {100 * hi2:.2f}\% for $\tau=2\%$ "
            rf"and from {100 * lo3:.2f}\% to {100 * hi3:.2f}\% for $\tau=3\%$.",
        ),
    ]
    for pattern, replacement in parts:
        text, count = re.subn(pattern, lambda _: replacement, text)
        if count != 1:
            raise ValueError(f"Expected one Appendix C risk-calibration anchor, found {count}")
    return text


def _bootstrap(net, rf, specs):
    """Use identical resampled months; Holm families are the prespecified comparisons."""
    excess = net.to_numpy() - rf.to_numpy()[:, None]
    rng = np.random.default_rng(20261002)
    n = len(rf)
    draws = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=(n + 5) // 6)
        sample = excess[((starts[:, None] + np.arange(6)) % n).ravel()[:n]]
        draws.append(np.sqrt(12) * sample.mean(0) / sample.std(0, ddof=1))
    draws = np.array(draws)
    sr = np.sqrt(12) * excess.mean(0) / excess.std(0, ddof=1)
    rows = []

    def pair(a, b, family):
        i, j = net.columns.get_loc(a), net.columns.get_loc(b)
        d = draws[:, i] - draws[:, j]
        low, high = np.quantile(d, [0.025, 0.975])
        beta, t, p = hac_regression(net[a] - net[b])
        rows.append(
            dict(
                variant=a,
                baseline=b,
                family=family,
                delta_sharpe=sr[i] - sr[j],
                ci_low=low,
                ci_high=high,
                bootstrap_p=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())),
                annual_mean_delta=12 * beta[0],
                hac_t=t[0],
                hac_p=p[0],
            )
        )

    for spec in specs:
        if spec.role == "additional":
            pair(spec.name, "source_weight", spec.family + "_vs_vw")
        if spec.family in ("match_bvc", "match_c", "tracking_error", "diversification"):
            pair(spec.name, f"{spec.kind}_36_2K", spec.family + "_vs_gmvp")
        if spec.family == "short_window":
            pair(
                spec.name,
                "lw_36_2K",
                "short_vs_lw36" if spec.role == "additional" else "ewma_check",
            )
    for window in (36, 60):
        for cap in CAPS:
            pair(f"lw_{window}_{cap}", f"sample_{window}_{cap}", "lw_vs_sample")
        for cap in ("2K", "none"):
            for a, b in (
                ("ff5bab", "ff5"),
                ("ff5bab", "lw"),
                ("ff5bab", "sample"),
                ("ff5", "lw"),
                ("ff5", "sample"),
            ):
                pair(f"{a}_{window}_{cap}", f"{b}_{window}_{cap}", "factor_comparison_" + cap)
    tests = pd.DataFrame(rows)
    for col in ("hac_p", "bootstrap_p"):
        tests["holm_" + col] = tests.groupby("family")[col].transform(
            lambda p: holm_adjust(p.to_numpy())
        )
    return tests


def run():
    out = OUTPUT / "revisions"
    out.mkdir(parents=True, exist_ok=True)
    panel = load_panel(Config())
    selection = load_selection(DATA / "selections/defensive_selection.csv", 30)
    months = pd.PeriodIndex(sorted(selection.holding_month.unique()), freq="M")
    ids = pd.Index(sorted(selection.PERMNO.unique()))
    source_sic = _raw_industries(selection, panel["sic"])
    bab = pd.read_csv(ROOT / "extensions/external_data/aqr_bab_usa.csv")
    factors = panel.factors.assign(
        BAB=pd.Series(bab.BAB_USA.to_numpy(), index=pd.PeriodIndex(bab.month, freq="M"))
    )
    specs = catalogue()
    solvers = _solvers()
    # The score vectors and every covariance estimate predate the holding-month return.
    realized = pd.DataFrame(0.0, index=months, columns=ids)
    targets = {s.name: realized.copy() for s in specs}
    ended = panel["delisted"].loc[months, ids]
    diagnostics = []
    for m, group in selection.groupby("holding_month"):
        names = pd.Index(group.PERMNO)
        r = panel["total_return"].loc[m, names]
        if not np.isfinite(r).all():
            raise ValueError(f"Incomplete holding returns: {m}")
        realized.loc[m, names] = r
        before = panel["total_return"].loc[: m - 1, names]
        missing = np.flatnonzero(before.isna().any(axis=1).to_numpy())
        before = before.iloc[missing[-1] + 1 :] if len(missing) else before
        if len(before) < 36:
            raise ValueError(f"Incomplete formation history: {m}")
        score = group[["factor_1", "factor_2", "factor_3"]].to_numpy()
        vw = validate_weights(
            module("factor_cov")._proj(group.weight.to_numpy(), CAP), names, CAP
        ).to_numpy()
        clean_sic = panel["sic"].loc[m - 1, names]
        sic = clean_sic.combine_first(source_sic.reindex(index=[m - 1], columns=names).iloc[0])
        # One explicit unknown industry for the manuscript's unfilled-metadata comparison.
        unfilled_sector = (clean_sic.fillna(-100).to_numpy() // 100).astype(int) + 1
        if sic.isna().any():
            raise ValueError(f"Unresolved formation industry: {m}")
        sector = (sic.to_numpy() // 100).astype(int)
        cache = {}
        for s in specs:
            key = (s.kind, s.window)
            h = before.iloc[-s.window :] if s.window else before
            if key not in cache:
                cache[key] = _covariance(s.kind, h, factors)
                if not np.isfinite(cache[key]).all():
                    raise ValueError(f"Nonfinite covariance at {m}, {key}")
            S = cache[key]
            w, info = _allocate(s, S, vw, score, sector, solvers)
            w = validate_weights(w, names, s.cap).to_numpy()
            targets[s.name].loc[m, names] = w
            totals = np.bincount(sector, weights=w)
            unfilled = np.bincount(unfilled_sector, weights=w)
            diagnostics.append(
                dict(
                    variant=s.name,
                    month=str(m),
                    months_used=len(h),
                    forecast_vol=np.sqrt(12 * w @ S @ w),
                    forecast_te=np.sqrt(max(0.0, 12 * (w - vw) @ S @ (w - vw))),
                    sector_hhi=totals @ totals,
                    sector_hhi_unfilled=unfilled @ unfilled,
                    largest_sector=totals.max(),
                    utilities_weight=totals[49] if len(totals) > 49 else 0.0,
                    top5=np.sort(w)[-5:].sum(),
                    relative_gap=info.get("relative_gap", np.nan),
                )
            )
        if len(diagnostics) // len(specs) % 50 == 0:
            print(
                f"Allocation sensitivity: {len(diagnostics) // len(specs)}/{len(months)} months",
                flush=True,
            )
    monthly = {}
    rows = []
    periods = {
        "full": (months[0], months[-1]),
        "2003_2011": ("2003-02", "2011-12"),
        "2012_2020": ("2012-01", "2020-12"),
        "crisis_ii": ("2020-02", "2020-06"),
    }
    for s in specs:
        pnl = evaluate(targets[s.name], realized, ended, PRIMARY_COST_BPS)
        pnl.index = months
        monthly[s.name] = pnl.net_return
        for period, (a, b) in periods.items():
            sub = pnl.loc[a:b]
            stats = performance(sub.net_return, panel.factors.RF)
            excess = sub.net_return - panel.factors.RF.loc[sub.index]
            stats.update(
                downside_deviation=np.sqrt(12 * np.minimum(excess, 0).pow(2).mean()),
                worst_5pct_mean=np.sort(sub.net_return)[: int(np.ceil(0.05 * len(sub)))].mean(),
            )
            rows.append(
                dict(
                    variant=s.name,
                    role=s.role,
                    family=s.family,
                    period=period,
                    mean_turnover=sub.turnover.mean(),
                    effective_n=sub.effective_n.mean(),
                    annual_gross_mean=12 * sub.gross_return.mean(),
                    **stats,
                )
            )
    net = pd.DataFrame(monthly)
    metrics = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    paired = _bootstrap(net, factors.RF.loc[months], specs)
    full = metrics.query("period=='full'").set_index("variant")
    risk = diag.groupby("variant", sort=False)[
        [
            "forecast_vol",
            "forecast_te",
            "sector_hhi",
            "sector_hhi_unfilled",
            "largest_sector",
            "top5",
            "utilities_weight",
        ]
    ].mean()
    risk["realized_vol"] = full.annual_volatility
    risk["realized_over_forecast"] = risk.realized_vol / risk.forecast_vol
    risk["realized_te"] = np.sqrt(12) * net.sub(net.source_weight, axis=0).std()

    def endpoints(values):
        return [float(values.min()), float(values.max())]

    archive_path = ROOT / "results/reference/monthly_returns.csv"
    archive = (
        pd.read_csv(archive_path)
        .query("scenario == 'total_observed_drift10'")
        .pivot(index="month", columns="model", values="net_return")
    )
    archive.index = pd.PeriodIndex(archive.index, freq="M")
    errors = {
        k: float(np.abs(net[k] - archive.loc[months, model]).max())
        for k, model in BASELINES.items()
    }
    if max(errors.values()) > 1e-7:
        raise ValueError(f"Baseline reproduction failed: {errors}")
    # The conditional downside experiment would require an improvement after score matching.
    restoration = []
    for kind in KINDS:
        baseline = f"{kind}_36_2K"
        for rule in ("match_bvc", "match_c"):
            variant = f"{kind}_{rule}"
            for metric in ("sharpe", "max_drawdown", "worst_5pct_mean", "downside_deviation"):
                difference = full.loc[variant, metric] - full.loc[baseline, metric]
                restoration.append(
                    dict(
                        variant=variant,
                        baseline=baseline,
                        metric=metric,
                        difference=difference,
                        improved=bool(
                            difference < 0 if metric == "downside_deviation" else difference > 0
                        ),
                    )
                )
    restoration = pd.DataFrame(restoration)
    restoration.to_csv(out / "allocation_score_matching.csv", index=False)
    concentration = {}
    for field in ("sector_hhi", "largest_sector", "utilities_weight", "top5"):
        pair = diag.pivot(index="month", columns="variant", values=field)
        coefficient, statistic, probability = hac_regression(
            pair["cc_36_2K"] - pair["source_weight"]
        )
        concentration[field] = dict(
            cc_minus_vw=coefficient[0], hac_t=statistic[0], hac_p=probability[0]
        )
    summary = dict(
        scope="49 additional settings, six baseline paths, separate 36-month EWMA diagnostic",
        forecast_definition="Full-sample annualized sample SD of monthly net returns / arithmetic mean month-ahead sqrt(12*w'Cov*w), using the covariance matrix used for allocation.",
        inference=dict(
            block_months=6,
            draws=2000,
            seed=20261002,
            hac_lags=4,
            holm="Within each reported comparison family",
        ),
        downside_condition=dict(
            comparisons=len(restoration), improvements=int(restoration.improved.sum())
        ),
        concentration_comparison=concentration,
        additional_settings=sum(s.role == "additional" for s in specs),
        months=len(months),
        risk_calibration=_risk_ranges(risk, specs),
        realized_te_limit_2pct=endpoints(risk.loc[[f"{k}_te2" for k in KINDS], "realized_te"]),
        realized_te_limit_3pct=endpoints(risk.loc[[f"{k}_te3" for k in KINDS], "realized_te"]),
        max_relative_objective_gap=float(diag.relative_gap.max()),
        baseline_max_abs_errors=errors,
    )
    metrics.to_csv(out / "allocation_metrics.csv", index=False)
    net.rename_axis("month").to_csv(out / "allocation_monthly_net_returns.csv")
    diag.to_csv(out / "allocation_monthly_diagnostics.csv.gz", index=False)
    risk.to_csv(out / "allocation_risk_calibration.csv")
    paired.to_csv(out / "allocation_paired_inference.csv", index=False)
    summary["table_rows_checked"] = {}
    manuscript = (PAPER / "main.tex").read_text()
    for key, body in _table_rows(metrics).items():
        (out / f"{key}.tex").write_text(body)
        summary["table_rows_checked"][key] = verify_table_rows(manuscript, key, body)
    (out / "allocation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Verified {summary['additional_settings']} additional settings, {len(months)} months, and both Appendix C tables.",
        flush=True,
    )


if __name__ == "__main__":
    run()
