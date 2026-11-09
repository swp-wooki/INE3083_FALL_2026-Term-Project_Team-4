"""Experiment E: factor-model covariance from FF5, with and without the BAB factor.

BAB is not in the project data; it comes from AQR's public series in external_data/. The
condition for this experiment held only without a position cap, so both cap settings run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from check_EA_conditions import bootstrap_difference, holm_by, sharpe
from common import (CAP, REFERENCE_MODEL, RESULTS, SOURCE, covariance, empty_targets, history, load_inputs,
                    module, publish, source_targets, validate_weights)
from ext01_window_cap import solve

EXP = "extE_factor_cov"
WINDOWS = [36, 60]
CAPS = {"cap2K": CAP, "capNone": 1.0}
FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def load_bab():
    bab = pd.read_csv(Path(__file__).parent / "external_data/aqr_bab_usa.csv")
    return pd.Series(bab.BAB_USA.to_numpy(), index=pd.PeriodIndex(bab.month, freq="M"), name="BAB")


def factor_covariance(h, factors):
    aligned = factors.loc[h.index]
    if aligned.isna().any().any():
        raise ValueError("Missing factor month")
    return module("factor_cov").factor_cov(h.sub(aligned.RF, axis=0), aligned.drop(columns="RF")).to_numpy()


def main():
    inp = load_inputs()
    factors = inp.panel.factors[[*FF5, "RF"]].copy()
    with_bab = factors.assign(BAB=load_bab().reindex(factors.index))
    specs = {"ff5": factors, "ff5bab": with_bab}
    variants = {SOURCE: dict(targets=source_targets(inp), role="baseline", method="market_cap",
                             reference_model=SOURCE)}
    rows = []
    for window in WINDOWS:
        for cap_name, cap in CAPS.items():
            # The same-setting statistical estimators that the factor models must beat.
            for kind in ["sample", "lw"]:
                name = f"{kind}_T{window}_{cap_name}"
                targets = empty_targets(inp)
                for m in inp.months:
                    w, _ = solve(covariance(kind, history(inp, m, window)), cap)
                    targets.loc[m, inp.names[m]] = validate_weights(w, inp.names[m], cap)
                variants[name] = dict(targets=targets, role="baseline", method=kind, window=window, cap=cap_name)
                if window == 36 and cap == CAP:
                    variants[name]["reference_model"] = REFERENCE_MODEL[kind]
            for label, frame in specs.items():
                name = f"{label}_T{window}_{cap_name}"
                print(name, flush=True)
                targets = empty_targets(inp)
                for m in inp.months:
                    S = factor_covariance(history(inp, m, window), frame)
                    w, gap = solve(S, cap)
                    targets.loc[m, inp.names[m]] = validate_weights(w, inp.names[m], cap)
                    rows.append(dict(variant=name, month=str(m), relative_gap=gap, condition=float(np.linalg.cond(S)),
                                     forecast_vol=float(np.sqrt(12 * w @ S @ w)), effective_n=float(1 / (w @ w))))
                report = label == "ff5" and window == 36 and cap == CAP
                variants[name] = dict(targets=targets, role="baseline" if report else "test", method=label,
                                      window=window, cap=cap_name, anchor=f"sample_T{window}_{cap_name}")
                if report:
                    variants[name]["reference_model"] = "gmvp_factor"
    diagnostics = pd.DataFrame(rows)
    series, _, _ = publish(
        EXP, 'Factor-model covariance (FF5 and FF5+BAB)',
        'Factor-model GMVPs use the packaged AQR U.S. monthly BAB series alongside FF5. Compare 36- and up-to-60-month windows at a 2/30 cap and without a stock cap. The matched anchor is sample GMVP.',
        inp, variants, dict(max_relative_gap=float(diagnostics.relative_gap.max()),
                            bab_source="AQR public series; see extensions/external_data"),
        {"solver_diagnostics.csv": diagnostics},
    )
    rf = inp.panel.factors.RF.reindex(inp.months).to_numpy()
    tests = []
    for window in WINDOWS:
        for cap_name in CAPS:
            for portfolio, other in [("ff5bab", "ff5"), ("ff5bab", "lw"), ("ff5bab", "sample"), ("ff5", "lw"),
                                     ("ff5", "sample")]:
                a, b = f"{portfolio}_T{window}_{cap_name}", f"{other}_T{window}_{cap_name}"
                x, y = series[a].net_return.to_numpy(), series[b].net_return.to_numpy()
                point, low, high, p = bootstrap_difference(x, y, rf, sharpe, True)
                tests.append(dict(family=cap_name, portfolio=a, compared_to=b, delta_sharpe=point, ci_low=low,
                                  ci_high=high, bootstrap_p=p,
                                  delta_annual_vol=float(np.sqrt(12) * (x.std(ddof=1) - y.std(ddof=1)))))
    tests = holm_by(pd.DataFrame(tests), "family")
    tests.to_csv(RESULTS / EXP / "estimator_comparisons.csv", index=False, encoding="utf-8-sig")
    forecast = diagnostics.groupby("variant")[["forecast_vol", "condition", "effective_n"]].mean()
    forecast["realized_vol"] = [float(np.sqrt(12) * series[v].net_return.std(ddof=1)) for v in forecast.index]
    forecast["realized_over_forecast"] = forecast.realized_vol / forecast.forecast_vol
    forecast.reset_index().to_csv(RESULTS / EXP / "forecast_vs_realized.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200, "display.max_rows", 100, "display.max_columns", 20)
    print(tests.round(4).to_string(index=False))
    print(forecast.round(4).to_string())


if __name__ == "__main__":
    main()
