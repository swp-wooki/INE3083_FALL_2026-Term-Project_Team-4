"""Conditional allocation experiment on the team's frozen low-risk/coskew selection."""

from .paths import DATA, OUTPUT, ROOT

P = OUTPUT
OUT = OUTPUT
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_panel
from .metrics import hac_regression, performance
from .models import MODELS, allocate, module, validate_weights
from .provenance import file_hash
from .selection import holm_adjust, load_selection

CAP = 2 / 30
PRIMARY = "total_observed_drift10"
PRIMARY_COST_BPS = 10
TC_COST_PARAMETER = 2 * PRIMARY_COST_BPS / 10000


def capped_proportional(v, cap=CAP):
    v = np.asarray(v, float)
    if not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("Positive finite capitalization required")
    w = np.zeros(len(v))
    free = np.ones(len(v), bool)
    while free.any():
        trial = v[free] / v[free].sum() * (1 - w.sum())
        over = trial > cap
        if not over.any():
            w[free] = trial
            break
        indices = np.flatnonzero(free)[over]
        w[indices] = cap
        free[indices] = False
    return w


def evaluate(targets, realized, ended, cost_bps, nominal=False):
    previous = pd.Series(0.0, index=targets.columns)
    rows = []
    for m in targets.index:
        w, r = targets.loc[m], realized.loc[m]
        traded = float((w - previous).abs().sum())
        gross = float(w @ r)
        cost = traded * cost_bps / 10000
        net = gross - cost if nominal else (1 - cost) * (1 + gross) - 1
        if not np.isfinite(net) or net <= -1:
            raise ValueError("Invalid portfolio return")
        rows.append(
            dict(
                month=str(m),
                gross_return=gross,
                net_return=net,
                turnover=traded / 2,
                traded_notional=traded,
                effective_n=1 / float(w @ w),
                max_weight=float(w.max()),
            )
        )
        previous = w.copy() if nominal else (w * (1 + r) / (1 + gross)).mask(ended.loc[m], 0.0)
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = DATA / "selections/defensive_selection.csv"
    s = load_selection(selection_path, 30)
    if (
        s.weight.isna().any()
        or not s.groupby("holding_month").weight.sum().between(0.99998, 1.00002).all()
    ):
        raise ValueError("Supplied weights outside rounding tolerance")
    cfg = Config()
    print("Loading local data", flush=True)
    panel = load_panel(cfg)
    months = pd.PeriodIndex(sorted(s.holding_month.unique()), freq="M")
    ids = pd.Index(sorted(s.PERMNO.unique()))
    raw = panel["total_return"]
    missing = raw.loc[months, ids].isna()
    ended = panel["delisted"].loc[months, ids]
    # Zeros only pad securities not selected that month; held returns must be observed.
    primary = pd.DataFrame(0.0, index=months, columns=ids)
    histories = {}
    names_by_month = {}
    audit = []
    for m, g in s.groupby("holding_month"):
        names = pd.Index(g.PERMNO)
        h = raw.loc[m - 36 : m - 1, names]
        if len(h) != 36 or not np.isfinite(h.to_numpy()).all() or h.std().le(1e-10).any():
            raise ValueError(f"Incomplete history {m}; no constituents substituted")
        realized = raw.loc[m, names]
        if not np.isfinite(realized.to_numpy()).all():
            raise ValueError(f"Unobserved selected return {m}; no return imputation permitted")
        primary.loc[m, names] = realized
        histories[m] = h
        names_by_month[m] = names
        for id in names:
            audit.append(
                dict(
                    formation_month=str(m - 1),
                    holding_month=str(m),
                    permno=int(id),
                    history_n=36,
                    missing_return=bool(missing.loc[m, id]),
                    security_end=bool(panel["delisted"].loc[m, id]),
                    total_return=panel["total_return"].loc[m, id],
                    price_return=panel["price_return"].loc[m, id],
                    sic=panel["sic"].loc[m - 1, id],
                    size=panel["size"].loc[m - 1, id],
                )
            )
    audit = pd.DataFrame(audit)
    audit.to_csv(OUT / "data_audit.csv", index=False)
    audit[audit.missing_return | audit.security_end].to_csv(
        OUT / "terminal_events.csv", index=False
    )
    s.to_csv(OUT / "frozen_selection.csv", index=False)
    model_names = ("source_weight", *(m for m in MODELS if m != "value_weight"))
    manifest = dict(
        status="running",
        selection_file=str(selection_path.relative_to(ROOT)),
        selection_sha256=file_hash(selection_path),
        months=len(months),
        rows=len(s),
        n=30,
        cap=CAP,
        window=36,
        models=model_names,
        primary=PRIMARY,
        missing_held_returns=int(audit.missing_return.sum()),
        scope="Frozen supplied selection; upstream daily beta, full score universe, and selector code unavailable. No K search or new factor discovery.",
        primary_accounting="Observed CRSP total returns, including restored original records; no price-return fallback or return imputation; drift-adjusted one-way 10bp (round-trip 20bp); terminal residual moves to cash; no return clipping",
        primary_one_way_cost_bps=PRIMARY_COST_BPS,
        round_trip_cost_bps=2 * PRIMARY_COST_BPS,
        sensitivity="Same target paths and observed returns under one-way costs 0, 10, 25, 50, 75 and 100bp; TC not retuned",
        source_weight="Six-decimal supplied capped value weights projected to sum one and 2/30 cap; all 30 names retained. Independent value-weight reconstruction omitted because some formation capitalizations are missing.",
        tc="gamma=5; cost parameter=0.002 (twice one-way 0.001); quadratic-form penalty lambda=0.0004; prior weights drifted under observed returns",
        data_provenance=panel.audit,
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
    )
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    weights = []
    returns = []
    source_adjustment = 0.0
    for model in model_names:
        print(model, flush=True)
        targets = pd.DataFrame(0.0, index=months, columns=ids)
        previous = pd.Series(dtype=float)
        for m in months:
            names, h = names_by_month[m], histories[m]
            if model == "source_weight":
                given = s[s.holding_month.eq(m)].set_index("PERMNO").weight.reindex(names)
                w = validate_weights(module("factor_cov")._proj(given.to_numpy(), CAP), names, CAP)
                source_adjustment = max(source_adjustment, float((w - given).abs().max()))
            elif model == "value_weight":
                w = validate_weights(
                    capped_proportional(panel["size"].loc[m - 1, names]), names, CAP
                )
            else:
                w = allocate(
                    model,
                    h,
                    panel["size"].loc[m - 1, names],
                    panel.factors.loc[h.index],
                    previous,
                    CAP,
                    TC_COST_PARAMETER,
                )
            targets.loc[m, names] = w
            weights.append(
                pd.DataFrame(
                    dict(
                        model=model,
                        holding_month=str(m),
                        permno=names,
                        weight=w.to_numpy(),
                    )
                )
            )
            r = primary.loc[m, names]
            gross = float(w @ r)
            previous = (w * (1 + r) / (1 + gross)).mask(ended.loc[m, names], 0.0)
        scenarios = {
            f"total_observed_drift{cost}": (primary, cost, False)
            for cost in [10, 0, 25, 50, 75, 100]
        }
        for scenario, (r, c, nominal) in scenarios.items():
            x = evaluate(targets, r, ended, c, nominal)
            x["model"] = model
            x["scenario"] = scenario
            x["missing_return_weight"] = (targets * missing).sum(axis=1).to_numpy()
            returns.append(x)
    positions = pd.concat(weights, ignore_index=True)
    returns = pd.concat(returns, ignore_index=True)
    periods = {
        "full": (str(months[0]), str(months[-1])),
        "2003_2011": ("2003-02", "2011-12"),
        "2012_2020": ("2012-01", "2020-12"),
        "financial_crisis": ("2007-07", "2009-03"),
        "covid": ("2020-02", "2020-06"),
    }
    metrics = []
    paired = []
    for (scenario, model), g in returns.groupby(["scenario", "model"]):
        g = g.set_index("month")
        g.index = pd.PeriodIndex(g.index, freq="M")
        for period, (a, b) in periods.items():
            sub = g.loc[a:b]
            stats = performance(sub.net_return, panel.factors.RF, panel.market, panel.factors)
            metrics.append(
                dict(
                    scenario=scenario,
                    model=model,
                    period=period,
                    **stats,
                    mean_turnover=sub.turnover.mean(),
                    mean_effective_n=sub.effective_n.mean(),
                )
            )
    metrics = pd.DataFrame(metrics)
    for scenario in returns.scenario.unique():
        wide = returns[returns.scenario.eq(scenario)].pivot(
            index="month", columns="model", values="net_return"
        )
        for period, (a, b) in periods.items():
            w = wide.loc[a:b]
            for baseline in ["source_weight", "equal_weight"]:
                for model in model_names:
                    if model == baseline:
                        continue
                    beta, t, p = hac_regression(w[model] - w[baseline], lags=4)
                    paired.append(
                        dict(
                            scenario=scenario,
                            period=period,
                            baseline=baseline,
                            model=model,
                            monthly_difference=beta[0],
                            annualized_difference=12 * beta[0],
                            hac_t=t[0],
                            hac_p=p[0],
                        )
                    )
    paired = pd.DataFrame(paired)
    paired["holm_p"] = paired.groupby(["scenario", "period", "baseline"]).hac_p.transform(
        lambda x: holm_adjust(x.to_numpy())
    )
    bm = pd.DataFrame(
        {
            "CRSP_market": panel.market.loc[months],
            "FF_market": (panel.factors["Mkt-RF"] + panel.factors.RF).loc[months],
            "RF": panel.factors.RF.loc[months],
        }
    )
    bmstats = []
    for name in ["CRSP_market", "FF_market"]:
        for period, (a, b) in periods.items():
            bmstats.append(dict(model=name, period=period, **performance(bm[name].loc[a:b], bm.RF)))
    # Check the saved holdings and independently reconstruct the primary return accounting.
    groups = positions.groupby(["holding_month", "model"])
    assert groups.size().eq(30).all() and np.allclose(groups.weight.sum(), 1)
    assert positions.weight.between(-1e-9, CAP + 1e-6).all()
    for (m, model), g in groups:
        assert set(g.permno) == set(names_by_month[pd.Period(m, "M")])
    for model in model_names:
        p = (
            positions[positions.model.eq(model)]
            .pivot(index="holding_month", columns="permno", values="weight")
            .fillna(0)
        )
        p.index = pd.PeriodIndex(p.index, freq="M")
        r = primary.reindex(columns=p.columns)
        gross = (p * r).sum(axis=1)
        actual = returns[returns.model.eq(model) & returns.scenario.eq(PRIMARY)]
        np.testing.assert_allclose(gross, actual.gross_return, atol=1e-12)
        np.testing.assert_allclose(
            (1 - PRIMARY_COST_BPS / 10000 * actual.traded_notional) * (1 + gross.to_numpy()) - 1,
            actual.net_return,
            atol=1e-12,
        )
    ew = np.array([primary.loc[m, names_by_month[m]].mean() for m in months])
    np.testing.assert_allclose(
        ew,
        returns.query('model == "equal_weight" and scenario == @PRIMARY').gross_return,
        atol=1e-12,
    )
    assert file_hash(selection_path) == manifest["selection_sha256"]
    positions.to_csv(OUT / "holdings.csv.gz", index=False)
    returns.to_csv(OUT / "monthly_returns.csv", index=False)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    paired.to_csv(OUT / "paired_comparisons.csv", index=False)
    bm.rename_axis("month").to_csv(OUT / "benchmark_returns.csv")
    pd.DataFrame(bmstats).to_csv(OUT / "benchmark_metrics.csv", index=False)
    annual = (
        returns.assign(year=returns.month.str[:4])
        .groupby(["scenario", "model", "year"])
        .net_return.agg(lambda x: (1 + x).prod() - 1)
    )
    annual.rename("annual_return").to_csv(OUT / "annual_returns.csv")
    # Reproducibility includes the immutable input, exact model code, and local data versions.
    files = [
        Path(__file__),
        selection_path,
        *sorted((Path(__file__).parent / "optimizers").glob("*.py")),
        *[
            Path(__file__).with_name(name + ".py")
            for name in ["config", "data", "metrics", "models", "paths", "provenance", "selection"]
        ],
        *[
            cfg.data_dir / n
            for n in [
                "raw/security_prices.csv.gz",
                "processed/security_monthly_clean.csv.gz",
                "raw/security_monthly.csv.gz",
                "raw/factor_returns_monthly.csv",
                "raw/market_benchmarks.csv",
            ]
        ],
    ]
    manifest.update(
        status="complete",
        source_weight_max_adjustment=source_adjustment,
        verification=dict(
            model_months=len(groups),
            positions=len(positions),
            same_constituents=True,
            weights_valid=True,
            gross_and_net_reconciled=True,
            equal_weight_independent_check=True,
            source_hash_unchanged=True,
        ),
        hashes={str(p.relative_to(ROOT)): file_hash(p) for p in files},
    )
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        metrics.query('scenario == @PRIMARY and period == "full"')[
            ["model", "cagr", "sharpe", "max_drawdown"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run()
