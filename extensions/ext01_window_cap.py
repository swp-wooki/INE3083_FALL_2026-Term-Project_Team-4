"""Experiment 1: estimation window x position cap for sample and Ledoit-Wolf GMVP.

Fixed stocks. Windows 24/36/60 months and caps 2/K, 3/K, none (still long only, fully
invested). Sample GMVP skips 24 months because its covariance is singular when T < N.
"""

import numpy as np
import pandas as pd

from common import (CAP, N, REFERENCE_MODEL, RESULTS, SOURCE, covariance, empty_targets,
                    frank_wolfe_gap, history, load_inputs, module, publish, source_targets,
                    validate_weights)

EXP = "ext01_window_cap"
CAPS = {"cap2K": 2 / N, "cap3K": 3 / N, "capNone": 1.0}
GRID = [("sample", 36), ("sample", 60), ("lw", 24), ("lw", 36), ("lw", 60)]
REFINE_ABOVE = 1e-7


def solve(S, cap):
    w = module("factor_cov").gmvp(S, cap=cap)
    gap = frank_wolfe_gap(S, w, cap)
    if gap > REFINE_ABOVE:
        # Wider caps and short windows converge more slowly; warm-start with more iterations.
        w = module("factor_cov").gmvp(S, cap=cap, w0=w, iters=400000, tol=1e-13)
        gap = frank_wolfe_gap(S, w, cap)
    if gap > 1e-6:
        raise RuntimeError(f"GMVP did not converge: gap={gap}")
    return w, gap


def main():
    inp = load_inputs()
    variants = {SOURCE: dict(targets=source_targets(inp), role="baseline", method="market_cap",
                             reference_model=SOURCE)}
    windows, solver_rows = [], []
    for kind, window in GRID:
        for cap_name, cap in CAPS.items():
            name = f"{kind}_T{window}_{cap_name}"
            print(name, flush=True)
            targets = empty_targets(inp)
            for m in inp.months:
                h = history(inp, m, window)
                S = covariance(kind, h)
                w, gap = solve(S, cap)
                targets.loc[m, inp.names[m]] = validate_weights(w, inp.names[m], cap)
                solver_rows.append(dict(variant=name, month=str(m), months_used=len(h), relative_gap=gap,
                                        condition=np.linalg.cond(S)))
            is_report = window == 36 and cap == CAP
            variants[name] = dict(targets=targets, role="baseline" if is_report else "test", method=kind,
                                  window=window, cap=cap_name, anchor=f"{kind}_T36_cap2K")
            if is_report:
                variants[name]["reference_model"] = REFERENCE_MODEL[kind]
    diagnostics = pd.DataFrame(solver_rows)
    used = diagnostics[diagnostics.variant.str.contains("T60")].groupby("month").months_used.first()
    checks = dict(
        max_relative_gap=float(diagnostics.relative_gap.max()),
        window60_months_used=dict(mean=float(used.mean()), min=int(used.min()), max=int(used.max()),
                                  months_with_full_60=int(used.eq(60).sum())),
    )
    series, metrics, paired = publish(
        EXP, "추정 기간 × 비중 상한 (표본·LW GMVP)",
        "종목은 고정하고 추정 기간 24/36/60개월과 비중 상한 2/K·3/K·무상한을 교차해 표본 GMVP와 "
        "LW(항등행렬 축소) GMVP를 다시 실행했습니다. 24개월은 T<N이라 LW만 실행했습니다. "
        "무상한도 공매도 금지와 비중 합 1은 유지합니다. 60개월은 30종목 공통 이력이 부족한 달에 "
        "가능한 최장 기간(최소 37개월)을 씁니다.",
        inp, variants, checks, {"solver_diagnostics.csv": diagnostics},
    )
    # A strict 60-month comparison exists only in the months where all 30 names have full history.
    strict = used[used.eq(60)].index
    net = pd.DataFrame({k: x.net_return for k, x in series.items()})
    net.index = net.index.astype(str)
    rf = inp.panel.factors.RF.copy()
    rf.index = rf.index.astype(str)
    rows = []
    for name in net.columns:
        r = net.loc[strict, name]
        excess = r - rf.reindex(strict)
        rows.append(dict(variant=name, months=len(r), annual_mean_return=12 * r.mean(),
                         annual_volatility=np.sqrt(12) * r.std(ddof=1),
                         sharpe=np.sqrt(12) * excess.mean() / excess.std(ddof=1)))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / EXP / "strict_60_month_subsample.csv", index=False, encoding="utf-8-sig")
    print(frame.round(4).to_string(index=False))
    print(checks)


if __name__ == "__main__":
    main()
