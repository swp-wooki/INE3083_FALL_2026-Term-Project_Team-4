"""Recompute portfolio diagnostics from the current verified allocation results."""

from .paths import OUTPUT

P = OUTPUT
OUT = OUTPUT
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import Config
from .data import load_panel
from .experiment import CAP, PRIMARY, PRIMARY_COST_BPS, evaluate
from .metrics import performance
from .models import module
from .provenance import file_hash


def improve_ratio(S, mu, start):
    def f(w):
        v = float(w @ S @ w)
        return -float(mu @ w) / np.sqrt(v)

    def g(w):
        v = float(w @ S @ w)
        return -mu / np.sqrt(v) + (mu @ w) * (S @ w) / v**1.5

    results = [
        minimize(
            f,
            x,
            jac=g,
            bounds=[(0, CAP)] * len(start),
            constraints={
                "type": "eq",
                "fun": lambda w: w.sum() - 1,
                "jac": lambda w: np.ones(len(w)),
            },
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        for x in [start, np.full(len(start), 1 / len(start))]
    ]
    valid = [
        r
        for r in results
        if r.success and abs(r.x.sum() - 1) < 1e-7 and r.x.min() > -1e-8 and r.x.max() < CAP + 1e-7
    ]
    if not valid:
        raise ValueError("Both ratio solver starts failed: " + str([r.message for r in results]))
    r = min(valid, key=lambda r: r.fun)
    if r.fun - f(start) > 1e-8:
        raise ValueError(f"Reference solver worsened objective by {r.fun - f(start)}")
    if f(start) < r.fun:
        return start.copy(), -f(start), -f(start)
    return r.x, -f(start), -r.fun


def run(check_local_ratios=False):
    s = pd.read_csv(P / "frozen_selection.csv")
    s["holding_month"] = pd.PeriodIndex(s.holding_month, freq="M")
    h = pd.read_csv(P / "holdings.csv.gz")
    h["holding_month"] = pd.PeriodIndex(h.holding_month, freq="M")
    cfg = Config()
    panel = load_panel(cfg)
    raw = panel["total_return"]
    months = pd.PeriodIndex(sorted(s.holding_month.unique()), freq="M")
    ids = pd.Index(sorted(s.PERMNO.unique()))
    primary = pd.DataFrame(0.0, index=months, columns=ids)
    ended = panel["delisted"].loc[months, ids]
    for month, sel in s.groupby("holding_month"):
        names = pd.Index(sel.PERMNO)
        observed = raw.loc[month, names]
        if not np.isfinite(observed.to_numpy()).all():
            raise ValueError(f"Unobserved selected return {month}")
        primary.loc[month, names] = observed
    out = []
    structure = []
    solve = []
    refs = {
        m: pd.DataFrame(0.0, index=months, columns=ids)
        for m in (["mdp", "max_sharpe_bs"] if check_local_ratios else [])
    }
    for ix, (month, sel) in enumerate(s.groupby("holding_month")):
        names = pd.Index(sel.PERMNO)
        X = raw.loc[month - 36 : month - 1, names].to_numpy()
        if X.shape != (36, len(names)) or not np.isfinite(X).all():
            raise ValueError(f"Incomplete observed history {month}")
        S = np.cov(X, rowvar=False)
        sd = np.sqrt(np.diag(S))
        corr = S / np.outer(sd, sd)
        ev = np.linalg.eigvalsh(S)
        f = panel.factors.loc[month - 36 : month - 1]
        market = f["Mkt-RF"].to_numpy()
        betas = np.array(
            [
                np.cov(X[:, i] - f.RF.to_numpy(), market, ddof=1)[0, 1] / np.var(market, ddof=1)
                for i in range(30)
            ]
        )
        off = corr[np.triu_indices(30, 1)]
        score = sel.set_index("PERMNO").loc[names, ["factor_1", "factor_2", "factor_3"]].to_numpy()
        ws = (
            h[h.holding_month.eq(month)]
            .pivot(index="model", columns="permno", values="weight")
            .reindex(columns=names)
        )
        source = ws.loc["source_weight"].to_numpy()
        sourcep = source @ score
        structure.append(
            dict(
                month=str(month),
                mean_correlation=off.mean(),
                correlation_dispersion=off.std(),
                covariance_condition=ev[-1] / ev[0],
                largest_eigenvalue_share=ev[-1] / ev.sum(),
                volatility_p90_p10=np.quantile(sd, 0.9) / np.quantile(sd, 0.1),
                selected_score1_score2=pd.Series(score[:, 0]).corr(
                    pd.Series(score[:, 1]), method="spearman"
                ),
                selected_score1_score3=pd.Series(score[:, 0]).corr(
                    pd.Series(score[:, 2]), method="spearman"
                ),
                selected_score2_score3=pd.Series(score[:, 1]).corr(
                    pd.Series(score[:, 2]), method="spearman"
                ),
            )
        )
        for model, wser in ws.iterrows():
            w = wser.to_numpy()
            rc = w * (S @ w) / (w @ S @ w)
            sc = w @ score
            out.append(
                dict(
                    month=str(month),
                    model=model,
                    effective_n=1 / (w @ w),
                    cap_count=int((w > CAP - 1e-6).sum()),
                    zero_count=int((w < 1e-7).sum()),
                    distance_source=0.5 * np.abs(w - source).sum(),
                    sample_fitted_vol=np.sqrt(12 * w @ S @ w),
                    weighted_asset_vol=np.sqrt(12) * (w @ sd),
                    fitted_beta=w @ betas,
                    estimated_monthly_mean=w @ X.mean(0),
                    rc_dispersion=np.std(rc),
                    weighted_score1=sc[0],
                    weighted_score2=sc[1],
                    weighted_score3=sc[2],
                    tilt_score1=sc[0] - sourcep[0],
                    tilt_score2=sc[1] - sourcep[1],
                    tilt_score3=sc[2] - sourcep[2],
                )
            )
        for model in refs:
            cov = S if model == "mdp" else module("cov_lw_identity").estimate(X)
            mu = (
                sd
                if model == "mdp"
                else module("max_sharpe_bs").bayes_stein(X, cov) - float(f.RF.iloc[-1])
            )
            w, old, new = improve_ratio(cov, mu, ws.loc[model].to_numpy())
            refs[model].loc[month, names] = w
            solve.append(
                dict(
                    month=str(month),
                    model=model,
                    legacy_objective=old,
                    reference_objective=new,
                    objective_gain=new - old,
                    half_l1_change=0.5 * np.abs(w - ws.loc[model]).sum(),
                    all_expected_excess_nonpositive=bool((mu <= 0).all()),
                )
            )
        if ix % 50 == 0:
            print("Audited", ix + 1, "months", flush=True)
    d = pd.DataFrame(out)
    st = pd.DataFrame(structure)
    sv = pd.DataFrame(solve)
    d.to_csv(P / "mechanism_monthly.csv", index=False)
    st.to_csv(P / "selected_covariance_structure.csv", index=False)
    if check_local_ratios:
        sv.to_csv(P / "solver_audit.csv", index=False)
    summary = d.groupby("model").mean(numeric_only=True)
    r = pd.read_csv(P / "monthly_returns.csv")
    primary_r = r[r.scenario.eq(PRIMARY)]
    decomp = []
    for model, g in primary_r.groupby("model"):
        v = summary.loc[model].to_dict()
        v.update(
            model=model,
            annual_gross_mean=12 * g.gross_return.mean(),
            annual_net_mean=12 * g.net_return.mean(),
            annual_cost_drag=12 * (g.gross_return - g.net_return).mean(),
            mean_turnover=g.turnover.mean(),
        )
        decomp.append(v)
    decomp = pd.DataFrame(decomp)
    base = decomp.set_index("model").loc["source_weight"]
    for key in ["annual_gross_mean", "annual_net_mean", "annual_cost_drag"]:
        decomp["delta_" + key] = decomp[key] - base[key]
    np.testing.assert_allclose(
        decomp.delta_annual_net_mean,
        decomp.delta_annual_gross_mean - decomp.delta_annual_cost_drag,
        atol=1e-12,
    )
    decomp.to_csv(P / "mechanism_summary.csv", index=False)
    ref_returns = []
    ref_metrics = []
    for model, weights in refs.items():
        for scenario, rr, cost in [
            (PRIMARY, primary, PRIMARY_COST_BPS),
            ("total_observed_drift0", primary, 0),
        ]:
            result = evaluate(weights, rr, ended, cost)
            result["model"] = model + "_slsqp"
            result["scenario"] = scenario
            ref_returns.append(result)
            series = result.set_index("month").net_return
            series.index = pd.PeriodIndex(series.index, freq="M")
            ref_metrics.append(
                dict(
                    model=model + "_slsqp",
                    scenario=scenario,
                    **performance(series, panel.factors.RF, panel.market, panel.factors),
                    mean_turnover=result.turnover.mean(),
                )
            )
    if check_local_ratios:
        pd.concat(ref_returns).to_csv(P / "solver_reference_returns.csv", index=False)
        pd.DataFrame(ref_metrics).to_csv(P / "solver_reference_metrics.csv", index=False)
    (P / "mechanism_manifest.json").write_text(
        json.dumps(
            dict(
                status="complete",
                months=len(months),
                model_months=len(d),
                input_hashes={
                    f: file_hash(P / f)
                    for f in [
                        "holdings.csv.gz",
                        "frozen_selection.csv",
                        "monthly_returns.csv",
                    ]
                },
                script_sha256=file_hash(Path(__file__)),
                scope="Post-hoc diagnostic; no upstream counterfactual selections or primary results changed",
                checks=[
                    "All held returns and estimation inputs observed; no return imputation",
                    "gross minus cost decomposition",
                ],
                local_ratio_comparison=check_local_ratios,
            ),
            indent=2,
        )
    )
    print(
        decomp[
            [
                "model",
                "effective_n",
                "mean_turnover",
                "tilt_score1",
                "tilt_score2",
                "tilt_score3",
                "delta_annual_gross_mean",
                "delta_annual_cost_drag",
                "delta_annual_net_mean",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(st.describe().loc[["mean", "50%"]].to_string())
    if check_local_ratios:
        print(
            pd.DataFrame(ref_metrics)[
                ["model", "scenario", "cagr", "sharpe", "mean_turnover"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    run()
