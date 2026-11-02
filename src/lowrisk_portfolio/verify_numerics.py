"""Verify revised saved experiment outputs without changing historical audit artifacts.

Independent formulae/certificates reuse pure helpers from audit_helpers;
its run/supplement/report entry points are deliberately never called.
"""

from .paths import OUTPUT, ROOT

P = OUTPUT
OUT = OUTPUT
import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from . import audit_helpers as ref

TARGET = OUT / "revision_verification.json"
PRIMARY = "total_observed_drift10"
PRIMARY_COST_RATE = 0.001
CAP = 2 / 30
KINDS = [
    "sample",
    "constant_corr",
    "lw_constant_corr",
    "lw_identity",
    "rmt_clipping",
    "qis",
    "ridge",
    "factor",
]
PERIODS = {
    "full": ("2003-02", "2020-12"),
    "2003_2011": ("2003-02", "2011-12"),
    "2012_2020": ("2012-01", "2020-12"),
    "financial_crisis": ("2007-07", "2009-03"),
    "covid": ("2020-02", "2020-06"),
}


def metric_values(x, rf):
    x = np.asarray(x, float)
    e = x - np.asarray(rf, float)
    wealth = np.r_[1.0, np.cumprod(1 + x)]
    return {
        "months": len(x),
        "cumulative_return": wealth[-1] - 1,
        "cagr": np.expm1(np.log1p(x).sum() * 12 / len(x)),
        "annual_mean_return": x.mean() * 12,
        "report_annual_return": (1 + x.mean()) ** 12 - 1,
        "annual_volatility": np.std(x, ddof=1) * np.sqrt(12),
        "sharpe": e.mean() / np.std(e, ddof=1) * np.sqrt(12),
        "max_drawdown": np.min(wealth / np.maximum.accumulate(wealth) - 1),
        "positive_month_fraction": np.mean(x > 0),
    }


def hac_t(y, x=None, lags=4):
    # Direct T-by-T Bartlett kernel, independent of production lag accumulation.
    y = np.asarray(y, float)
    z = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(z, y, rcond=None)[0]
    u = y - z @ beta
    distance = np.abs(np.arange(len(y))[:, None] - np.arange(len(y))[None, :])
    kernel = np.maximum(0.0, 1 - distance / (lags + 1))
    bread = np.linalg.pinv(z.T @ z)
    scores = z * u[:, None]
    covariance = bread @ scores.T @ kernel @ scores @ bread
    t = beta / np.sqrt(np.diag(covariance))
    return beta, t, 2 * norm.sf(np.abs(t))


def tc_certificate(S, a, w, lam=0.0004):
    # A single LP yields a global convex upper bound at the actual SAVED iterate.
    n = len(w)
    scale = np.trace(S) / n
    Q = S / scale
    penalty = lam / scale
    z = np.r_[w, np.abs(w - a)]
    g = np.r_[2 * Q @ w, np.full(n, penalty)]
    A = np.block([[-np.eye(n), np.eye(n)], [np.eye(n), np.eye(n)]])
    b = np.r_[-a, a]
    budget = np.r_[np.ones(n), np.zeros(n)]
    answer = linprog(
        g,
        A_ub=-A,
        b_ub=-b,
        A_eq=[budget],
        b_eq=[1.0],
        bounds=[(0, CAP)] * n + [(0, 2)] * n,
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    if not answer.success:
        raise RuntimeError("TC certificate LP failed: " + answer.message)
    objective = w @ Q @ w + penalty * np.abs(w - a).sum()
    return max(0.0, float(g @ z - answer.fun)) / max(float(objective), 1e-30)


def attach_solvers(result):
    path = OUT / "revision_solver_verification.json"
    if not path.exists():
        result["ratio_erc_verification"] = {"status": "pending", "path": path.name}
        result["complete"] = False
        return
    value = json.loads(path.read_text())
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    holdings = pd.read_csv(OUT / "holdings.csv.gz")
    solver_checks = []

    def record(name, error, tolerance):
        solver_checks.append(
            {
                "check": name,
                "error": float(error),
                "tolerance": tolerance,
                "passed": bool(np.isfinite(error) and error <= tolerance),
            }
        )

    record("Certified monthly input count", abs(len(value["rows"]) - 215), 0)
    record(
        "Certificates use the current observed-return scenario",
        int(value.get("primary_scenario") != PRIMARY),
        0,
    )
    record(
        "Certificates identify the current saved holdings",
        int(value.get("holdings_sha256") != ref.digest(OUT / "holdings.csv.gz")),
        0,
    )
    record(
        "Certificates and saved holdings share inputs",
        sum(
            sha != manifest["hashes"].get(name, ref.digest(ROOT / name))
            for name, sha in value["input_hashes"].items()
        ),
        0,
    )
    record(
        "Certified solver code is current",
        sum(sha != ref.digest(ROOT / name) for name, sha in value["code_hashes"].items()),
        0,
    )
    maximum = {"mdp": 0.0, "max_sharpe_bs": 0.0, "erc": 0.0}
    positive_gap = []
    positive_feasibility = []
    negative_errors = []
    erc_kkt = []
    erc_budget = []
    for row in value["rows"]:
        block = holdings[holdings.holding_month.eq(row["month"])]
        for model in maximum:
            saved = (
                block[block.model.eq(model)]
                .set_index("permno")
                .weight.reindex(row["ids"])
                .to_numpy()
            )
            raw = np.asarray(row[model]["weights"], float)
            normalized = np.maximum(raw, 0.0)
            normalized /= normalized.sum()
            maximum[model] = max(maximum[model], float(np.max(abs(saved - normalized))))
            cert = row[model]
            if model == "erc":
                erc_kkt.append(cert["relative_inner_kkt_residual"])
                erc_budget.append(abs(cert["budget_residual"]))
            elif cert["method"] == "homogeneous_convex_qp":
                positive_gap.append(cert["relative_duality_gap"])
                positive_feasibility.append(cert["primal_residual"])
            else:
                negative_errors.append(abs(cert["exhaustive_reference_objective_difference"]))
    for model, error in maximum.items():
        record(
            "Saved " + model + " weights match normalized certified solution",
            error,
            1e-8,
        )
    record("Positive ratio duality certificates", max(positive_gap, default=0.0), 1e-8)
    record(
        "Positive ratio primal feasibility",
        max(positive_feasibility, default=0.0),
        1e-8,
    )
    record(
        "All negative-ratio certificates independently enumerated",
        abs(len(negative_errors) - value["negative_months_enumerated"]),
        0,
    )
    record(
        "Negative-ratio agreement with fresh exhaustive reference",
        max(negative_errors, default=0.0),
        1e-12,
    )
    record("ERC inner KKT certificates", max(erc_kkt), 1e-12)
    record("ERC pre-normalization budget certificates", max(erc_budget), 1e-10)
    passed = all(item["passed"] for item in solver_checks)
    result["ratio_erc_verification"] = {
        "path": path.name,
        "sha256": ref.digest(path),
        "holdings_sha256": ref.digest(OUT / "holdings.csv.gz"),
        "saved_holdings_verified": passed,
        "checks": solver_checks,
        "normalized_weight_difference": maximum,
        "source_scope": value.get("scope"),
        "numerical_tolerances": value["numerical_tolerances"],
        "assumptions": value["assumptions"],
    }
    for model in maximum:
        result["model_coverage"][model]["numerical_status"] = (
            "saved weights match certified revised solver; bounds and accounting verified"
            if passed
            else "inspect dedicated solver verification failures"
        )
    result["complete"] = result["passed"] and passed


def run():
    # macOS Accelerate may leave spurious matmul floating-point flags; explicit
    # finite-output, independent formula and residual checks determine failures.
    np.seterr(all="ignore")
    manifest_path = OUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("Experiment has not completed")
    if manifest.get("primary") != PRIMARY or manifest.get("primary_one_way_cost_bps") != 10:
        raise RuntimeError("Experiment does not use observed returns and one-way 10bp")
    mismatched = [
        name for name, sha in manifest["hashes"].items() if ref.digest(ROOT / name) != sha
    ]
    if mismatched:
        raise RuntimeError("Manifest is stale or inputs changed: " + ", ".join(mismatched))
    data, _ = ref.inputs()
    checks = []

    def check(name, error, tolerance):
        checks.append(
            {
                "check": name,
                "error": float(error),
                "tolerance": tolerance,
                "passed": bool(np.isfinite(error) and error <= tolerance),
            }
        )

    used = [
        "run_manifest.json",
        "holdings.csv.gz",
        "frozen_selection.csv",
        "monthly_returns.csv",
        "model_metrics.csv",
        "regime_metrics.csv",
        "regime_coverage.csv",
        "benchmark_returns.csv",
        "paired_comparisons.csv",
        "sharpe_bootstrap.csv",
    ]
    snapshot = {p: ref.digest(OUT / p) for p in used}
    holdings = pd.read_csv(OUT / "holdings.csv.gz")
    selection = pd.read_csv(OUT / "frozen_selection.csv")
    returns = pd.read_csv(OUT / "monthly_returns.csv")
    benchmark = pd.read_csv(OUT / "benchmark_returns.csv").set_index("month")
    models = manifest["models"]
    months = data["months"]
    allids = data["allids"]
    check("19 allocation rules", abs(len(models) - 19), 0)
    check("No imputed held returns", manifest["missing_held_returns"], 0)
    check(
        "No imputed return exposure in any saved scenario",
        returns.missing_return_weight.abs().max(),
        0,
    )
    check(
        "Only observed-return cost scenarios are present",
        len(set(returns.scenario) ^ {f"total_observed_drift{c}" for c in [0, 10, 25, 50, 75, 100]}),
        0,
    )
    check("215 monthly decisions", abs(len(months) - 215), 0)
    check("122550 saved positions", abs(len(holdings) - 215 * 19 * 30), 0)
    check(
        "No duplicate saved model-month-security keys",
        holdings.duplicated(["model", "holding_month", "permno"]).sum(),
        0,
    )
    check(
        "30 positions per model-month",
        np.max(abs(holdings.groupby(["model", "holding_month"]).size() - 30)),
        0,
    )
    check(
        "Selection source SHA256 preserved",
        int(ref.digest(ROOT / manifest["selection_file"]) != manifest["selection_sha256"]),
        0,
    )
    formation = pd.PeriodIndex(selection.formation_month, freq="M")
    holding = pd.PeriodIndex(selection.holding_month, freq="M")
    check(
        "Holding month follows decision month",
        np.max(abs(holding.asi8 - formation.asi8 - 1)),
        0,
    )
    check(
        "Selection has6450 distinct decision-security pairs",
        abs(len(selection) - 6450) + selection.duplicated(["holding_month", "PERMNO"]).sum(),
        0,
    )
    covariance_rows = []
    portfolio_rows = []
    previous_tc = pd.Series(dtype=float)
    for index, month in enumerate(months):
        X = data["X"][index]
        ids = data["ids"][index]
        f = pd.DataFrame(data["F"][index], columns=data["factor_names"])
        history = pd.DataFrame(X, columns=ids)
        saved = holdings[holdings.holding_month.eq(month)]
        wtable = saved.pivot(index="model", columns="permno", values="weight").reindex(
            index=models, columns=ids
        )
        check("Constituents " + month, len(set(saved.permno) - set(ids)), 0)
        check("Finite weights " + month, int(not np.isfinite(wtable.to_numpy()).all()), 0)
        S = np.cov(X, rowvar=False)
        sd = np.sqrt(np.diag(S))
        covariance = {}
        for kind in KINDS:
            if kind == "factor":
                C = (
                    ref.module("factor_cov")
                    .factor_cov(history.sub(f.RF, axis=0), f.drop(columns="RF"))
                    .to_numpy()
                )
                independent = ref.factor_ref(X, f)
            else:
                estimator = ref.module("cov_" + kind).estimate
                C = estimator(X, legacy=False) if kind == "lw_constant_corr" else estimator(X)
                if kind == "sample":
                    y = X - X.mean(0)
                    independent = y.T @ y / (len(X) - 1)
                elif kind == "constant_corr":
                    correlation = np.corrcoef(X, rowvar=False)
                    rho = correlation[np.triu_indices(len(ids), 1)].mean()
                    independent = rho * np.outer(sd, sd)
                    np.fill_diagonal(independent, sd * sd)
                elif kind == "lw_constant_corr":
                    independent = ref.lwcc_ref(X)
                elif kind == "lw_identity":
                    independent = LedoitWolf().fit(X).covariance_
                elif kind == "qis":
                    independent = ref.qis_ref(X)
                elif kind == "ridge":
                    independent = S + 0.3 * np.diag(np.repeat(np.diag(S).mean(), len(ids)))
                else:
                    correlation = np.corrcoef(X, rowvar=False)
                    e, U = np.linalg.eigh(correlation)
                    noise = e < (1 + np.sqrt(len(ids) / len(X))) ** 2
                    if noise.any():
                        e[noise] = e[noise].mean()
                    correlation = (U * e) @ U.T
                    scaled = sd / np.sqrt(correlation.diagonal())
                    independent = correlation * np.outer(scaled, scaled)
            if not np.isfinite(C).all() or not np.isfinite(independent).all():
                raise RuntimeError("Nonfinite covariance at " + month + "/" + kind)
            covariance[kind] = C
            covariance_rows.append(
                {
                    "month": month,
                    "kind": kind,
                    "relative_formula_error": float(
                        np.max(abs(C - independent)) / np.max(abs(independent))
                    ),
                    "minimum_eigenvalue": float(np.linalg.eigvalsh(C)[0]),
                    "symmetry_error": float(np.max(abs(C - C.T))),
                }
            )
        for model in models:
            w = wtable.loc[model].to_numpy()
            row = {
                "month": month,
                "model": model,
                "budget_error": abs(float(w.sum()) - 1),
                "bound_violation": max(0.0, -float(w.min()), float(w.max()) - CAP),
            }
            if model.startswith("gmvp_"):
                row["relative_convex_objective_error_bound"] = ref.gap(covariance[model[5:]], w)
            elif model == "tc_gmvp":
                a = previous_tc.reindex(ids, fill_value=0).to_numpy()
                row["relative_convex_objective_error_bound"] = tc_certificate(S, a, w)
                r = pd.Series(data["R"][index], index=allids).reindex(ids)
                ended = pd.Series(data["ended"][index], index=allids).reindex(ids)
                previous_tc = (pd.Series(w, index=ids) * (1 + r) / (1 + float(w @ r))).mask(
                    ended, 0.0
                )
            elif model == "min_cvar":
                row["absolute_primal_dual_gap"] = abs(ref.cvar_dual(X, w))
            elif model == "norm_gmvp":
                row["formula_error"] = float(np.max(abs(w - wtable.loc["gmvp_sample"].to_numpy())))
            elif model == "mix_1n":
                row["formula_error"] = float(
                    np.max(abs(w - (0.5 / len(ids) + 0.5 * wtable.loc["gmvp_sample"].to_numpy())))
                )
            elif model == "equal_weight":
                row["formula_error"] = float(np.max(abs(w - 1 / len(ids))))
            elif model == "source_weight":
                source = (
                    selection[selection.holding_month.eq(month)]
                    .set_index("PERMNO")
                    .weight.reindex(ids)
                    .to_numpy()
                )
                row["formula_error"] = float(np.max(abs(w - ref.projection(source, CAP))))
            elif model in ["inverse_vol", "vol_timing"]:
                base = 1 / (sd if model == "inverse_vol" else sd * sd)
                base /= base.sum()
                row["formula_error"] = float(np.max(abs(w - ref.projection(base, CAP))))
            portfolio_rows.append(row)
        if (index + 1) % 50 == 0:
            print(f"Covariance and certificates: {index + 1}/{len(months)}", flush=True)
    cov = pd.DataFrame(covariance_rows)
    portfolios = pd.DataFrame(portfolio_rows)
    check(
        "All1720 covariance formula comparisons",
        cov.relative_formula_error.max(),
        1e-12,
    )
    check(
        "Positive-semidefinite covariance inputs",
        max(0.0, -cov.minimum_eigenvalue.min()),
        1e-12,
    )
    check("Covariance symmetry", cov.symmetry_error.max(), 1e-12)
    check("All4085 budgets", portfolios.budget_error.max(), 1e-9)
    check("All4085 caps and nonnegativity", portfolios.bound_violation.max(), 1e-8)
    check(
        "All1720 GMVP global convex bounds",
        portfolios[
            portfolios.model.str.startswith("gmvp_")
        ].relative_convex_objective_error_bound.max(),
        1e-7,
    )
    check(
        "All215 TC global convex bounds at saved weights",
        portfolios[portfolios.model.eq("tc_gmvp")].relative_convex_objective_error_bound.max(),
        1e-6,
    )
    check("All215 CVaR primal-dual gaps", portfolios.absolute_primal_dual_gap.max(), 1e-9)
    check(
        "All215 norm allocations exactly duplicate sample GMVP",
        portfolios[portfolios.model.eq("norm_gmvp")].formula_error.max(),
        0,
    )
    check(
        "Source, equal, mixture, inverse-vol and VT formulae",
        portfolios[~portfolios.model.eq("norm_gmvp")].formula_error.max(),
        2e-9,
    )
    accounting = []
    primary = returns[returns.scenario.eq(PRIMARY)]
    for model in models:
        weights = (
            holdings[holdings.model.eq(model)]
            .pivot(index="holding_month", columns="permno", values="weight")
            .reindex(index=months, columns=allids)
            .fillna(0)
            .to_numpy()
        )
        prev = np.zeros(len(allids))
        net = []
        gross = []
        traded = []
        for index, w in enumerate(weights):
            amount = sum(abs(float(w[j] - prev[j])) for j in range(len(w)))
            g = sum(float(w[j] * data["R"][index, j]) for j in range(len(w)))
            net.append((1 - PRIMARY_COST_RATE * amount) * (1 + g) - 1)
            gross.append(g)
            traded.append(amount)
            prev = w * (1 + data["R"][index]) / (1 + g)
            prev[data["ended"][index]] = 0.0
        saved = primary[primary.model.eq(model)].set_index("month").loc[months]
        accounting.append(
            {
                "model": model,
                "net_error": float(np.max(abs(np.array(net) - saved.net_return))),
                "gross_error": float(np.max(abs(np.array(gross) - saved.gross_return))),
                "traded_error": float(np.max(abs(np.array(traded) - saved.traded_notional))),
            }
        )
    for item in ["net_error", "gross_error", "traded_error"]:
        check("All19 scalar ledger " + item, max(v[item] for v in accounting), 1e-12)
    metrics = pd.read_csv(OUT / "model_metrics.csv")
    metric_errors = []
    hac_errors = []
    ff3_errors = []
    # Factor windows already include the36 preceding months; load once for evaluation dates.
    from .data import read_factors

    factors = read_factors(ref.Config().data_dir / "raw/factor_returns_monthly.csv")
    for v in metrics.itertuples():
        low, high = PERIODS[v.period]
        g = returns[
            (returns.scenario == v.scenario)
            & (returns.model == v.model)
            & returns.month.between(low, high)
        ].sort_values("month")
        x = g.net_return.to_numpy()
        rf = benchmark.loc[g.month, "RF"].to_numpy()
        for key, value in metric_values(x, rf).items():
            metric_errors.append(abs(getattr(v, key) - value))
        _, t, p = hac_t(x - benchmark.loc[g.month, "CRSP_market"].to_numpy())
        hac_errors.extend([abs(t[0] - v.active_return_hac_t), abs(p[0] - v.active_return_hac_p)])
        idx = pd.PeriodIndex(g.month, freq="M")
        beta, t, p = hac_t(x - rf, factors.loc[idx, ["Mkt-RF", "SMB", "HML"]].to_numpy())
        ff3_errors.extend(
            [
                abs(12 * beta[0] - v.ff3_alpha_annual),
                abs(t[0] - v.ff3_alpha_hac_t),
                abs(p[0] - v.ff3_alpha_hac_p),
                abs(beta[1] - v.market_beta),
            ]
        )
    check("All saved scenario/period performance metrics", max(metric_errors), 1e-10)
    check("All saved active-return HAC statistics", max(hac_errors), 1e-8)
    check("All saved FF3 coefficient and HAC statistics", max(ff3_errors), 1e-8)
    cover = pd.read_csv(OUT / "regime_coverage.csv")
    counts = pd.Series(0, index=months)
    for v in cover.itertuples():
        used_months = counts.index[
            (counts.index >= v.observed_start) & (counts.index <= v.observed_end)
        ]
        counts.loc[used_months] += 1
        check("Regime " + v.regime + " month count", abs(len(used_months) - v.months), 0)
    check("Regimes partition215 months once", np.max(abs(counts - 1)), 0)
    pre = cover[cover.regime.eq("Pre-Crisis")].iloc[0]
    crisis = cover[cover.regime.eq("Crisis II")].iloc[0]
    check(
        "2020-02 belongs only to Crisis II",
        int(pre.observed_end != "2020-01" or crisis.observed_start != "2020-02"),
        0,
    )
    regime = pd.read_csv(OUT / "regime_metrics.csv")
    regime_errors = []
    for v in regime.itertuples():
        if v.scenario == "benchmark":
            x = benchmark.loc[v.start : v.end, v.model]
        else:
            g = returns[
                (returns.scenario == v.scenario)
                & (returns.model == v.model)
                & returns.month.between(v.start, v.end)
            ].sort_values("month")
            x = pd.Series(g.net_return.to_numpy(), index=g.month)
        for key, value in metric_values(x.to_numpy(), benchmark.loc[x.index, "RF"]).items():
            regime_errors.append(abs(getattr(v, key) - value))
    check("All saved regime metrics", max(regime_errors), 1e-10)
    paired = pd.read_csv(OUT / "paired_comparisons.csv")
    paired_errors = []
    holm_errors = []
    for v in paired.itertuples():
        low, high = PERIODS[v.period]
        group = returns[returns.scenario.eq(v.scenario) & returns.month.between(low, high)]
        wide = group.pivot(index="month", columns="model", values="net_return")
        d = (wide[v.model] - wide[v.baseline]).to_numpy()
        _, t, p = hac_t(d)
        paired_errors.extend(
            [
                abs(d.mean() - v.monthly_difference),
                abs(12 * d.mean() - v.annualized_difference),
                abs(t[0] - v.hac_t),
                abs(p[0] - v.hac_p),
            ]
        )
    for _, group in paired.groupby(["scenario", "period", "baseline"]):
        order = np.argsort(group.hac_p.to_numpy())
        ps = group.hac_p.to_numpy()[order]
        adjusted = np.minimum(1, np.maximum.accumulate(ps * np.arange(len(ps), 0, -1)))
        holm_errors.extend(abs(adjusted - group.holm_p.to_numpy()[order]))
    check("All paired comparison estimates and HAC", max(paired_errors), 1e-8)
    check("All Holm families", max(holm_errors), 1e-12)
    wide = primary.pivot(index="month", columns="model", values="net_return")
    ex = wide.to_numpy() - benchmark.loc[wide.index, "RF"].to_numpy()[:, None]
    T = len(ex)
    rng = np.random.default_rng(20261002)
    samples = []
    for _ in range(2000):
        starts = rng.integers(T, size=(T + 5) // 6)
        idx = np.array([(int(s) + j) % T for s in starts for j in range(6)])[:T]
        value = ex[idx]
        samples.append(value.mean(0) / value.std(0, ddof=1) * np.sqrt(12))
    samples = np.array(samples)
    bootstrap_errors = []
    empirical = ex.mean(0) / ex.std(0, ddof=1) * np.sqrt(12)
    for v in pd.read_csv(OUT / "sharpe_bootstrap.csv").itertuples():
        i = wide.columns.get_loc(v.model)
        j = wide.columns.get_loc(v.baseline)
        low, high = np.quantile(samples[:, i] - samples[:, j], [0.025, 0.975])
        bootstrap_errors.extend(
            [
                abs(low - v.ci_low),
                abs(high - v.ci_high),
                abs(empirical[i] - empirical[j] - v.delta_sharpe),
            ]
        )
        check(
            "Bootstrap settings " + v.model + " vs " + v.baseline,
            abs(v.replications - 2000) + abs(v.block_months - 6) + abs(v.seed - 20261002),
            0,
        )
    check("All paired circular-block bootstrap results", max(bootstrap_errors), 1e-10)
    check(
        "All audited output files unchanged during verification",
        sum(ref.digest(OUT / p) != sha for p, sha in snapshot.items()),
        0,
    )
    status = {
        name: {
            "numerical_status": "passed" if all(v["passed"] for v in checks) else "inspect_checks",
            "literature_status": "formula_verified_with_project_constraints",
        }
        for name in models
    }
    status["source_weight"]["literature_status"] = (
        "source weights reproduced; upstream selection/market-cap provenance incomplete"
    )
    for name in [
        "gmvp_rmt_clipping",
        "gmvp_ridge",
        "gmvp_factor",
        "inverse_vol",
        "vol_timing",
        "mix_1n",
    ]:
        status[name]["literature_status"] = (
            "explicit project variant, not full original-paper replication"
        )
    status["norm_gmvp"]["literature_status"] = (
        "redundant constraint; same problem as sample GMVP; A07 full-text verification incomplete"
    )
    for name in ["erc", "mdp", "max_sharpe_bs"]:
        status[name]["numerical_status"] = (
            "bounds/accounting verified; await dedicated solver attachment"
        )
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "months": len(months),
            "models": len(models),
            "model_months": len(portfolio_rows),
            "positions": len(holdings),
            "covariance_windows": len(covariance_rows),
        },
        "output_hashes": snapshot,
        "manifest_hash_match": True,
        "checks": checks,
        "passed": all(v["passed"] for v in checks),
        "model_coverage": status,
        "covariance_checks": covariance_rows,
        "portfolio_checks": portfolio_rows,
        "accounting": accounting,
        "limitations": [
            "Actual historical authorship/reading provenance is not established by equation agreement.",
            "Upstream daily beta, full candidate universe and selector code remain unverified.",
            "Observed total returns were recovered from supplied original CRSP records; no missing selected returns are imputed.",
            "A07 original full text was not readable in this review; norm KKT relation derived directly.",
            "Finite input-window certification does not prove arbitrary-input correctness or investment profitability.",
        ],
        "historical_audit_modified": False,
        "environment": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }
    attach_solvers(result)
    TARGET.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
    )
    failures = [v for v in checks if not v["passed"]]
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": len(checks),
                "failures": failures,
                "complete": result["complete"],
                "scope": result["scope"],
            },
            indent=2,
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attach-solvers", action="store_true")
    args = parser.parse_args()
    if args.attach_solvers:
        result = json.loads(TARGET.read_text())
        attach_solvers(result)
        TARGET.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        sys.exit(run())
