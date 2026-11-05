"""Experiment 9: does the FF5 alpha survive adding the BAB factor?

Regresses each report strategy's saved net excess returns on FF5 and on FF5+BAB, with the
report's Bartlett HAC(4) convention. No portfolio is rebuilt; returns are the supplied ones.
"""

import json
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from common import DATA, PRIMARY, RESULTS, ROOT, SOURCE, _git, _sha, holm_adjust, rebuild_ledger
from extE_factor_cov import load_bab
from lowrisk_portfolio.data import read_factors
from lowrisk_portfolio.market_inference import alpha_hac
from lowrisk_portfolio.metrics import hac_regression

EXP = "ext09_alpha_bab"
FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
SPECS = {"ff5": FF5, "ff5_bab": [*FF5, "BAB"]}
PERIODS = {"full": ("2003-02", "2020-12"), "2003_2011": ("2003-02", "2011-12"),
           "2012_2020": ("2012-01", "2020-12")}
MAIN = {"source_weight": "Market-cap weight", "equal_weight": "Equal weight",
        "gmvp_constant_corr": "CC GMVP", "gmvp_lw_identity": "LW identity"}


def regress(y, X):
    beta, t, p = hac_regression(y, X, lags=4)
    fitted = np.column_stack([np.ones(len(y)), X]) @ beta
    r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, t, p, float(r2)


def main():
    returns = pd.read_csv(ROOT / "results/reference/monthly_returns.csv")
    returns = returns[returns.scenario.eq(PRIMARY)].pivot(index="month", columns="model", values="net_return")
    factors = read_factors(DATA / "raw/factor_returns_monthly.csv")
    factors.index = factors.index.astype(str)
    bab = load_bab()
    bab.index = bab.index.astype(str)
    factors["BAB"] = bab.reindex(factors.index)
    reference = pd.read_csv(ROOT / "results/reference/market_inference.csv")
    reference = reference[reference.market.eq("FF_market") & reference.block_months.eq(6)].set_index("period")
    rows, checks = [], {}
    for period, (a, b) in PERIODS.items():
        months = returns.loc[a:b].index
        f = factors.loc[months]
        if f[SPECS["ff5_bab"]].isna().any().any():
            raise ValueError(f"Missing factor month in {period}")
        for model in returns.columns:
            excess = (returns.loc[months, model] - f.RF).to_numpy()
            row = dict(variant=model, period=period, months=len(months), role="baseline" if model == SOURCE else "test")
            for spec, columns in SPECS.items():
                beta, t, p, r2 = regress(excess, f[columns].to_numpy())
                row.update({f"{spec}_alpha_annual": 12 * beta[0], f"{spec}_alpha_hac_t": t[0],
                            f"{spec}_alpha_hac_p": p[0], f"{spec}_r2": r2})
                for name, b_, t_ in zip(columns, beta[1:], t[1:]):
                    row.update({f"{spec}_beta_{name}": b_, f"{spec}_t_{name}": t_})
                if spec == "ff5":
                    # The report's own estimator must agree with the shared HAC routine.
                    check = alpha_hac(excess, f[columns].to_numpy())
                    np.testing.assert_allclose(check, (12 * beta[0], t[0], p[0]), rtol=1e-10, atol=1e-14)
            row["alpha_change_annual"] = row["ff5_bab_alpha_annual"] - row["ff5_alpha_annual"]
            row["r2_change"] = row["ff5_bab_r2"] - row["ff5_r2"]
            rows.append(row)
        source = next(r for r in rows if r["variant"] == SOURCE and r["period"] == period)
        expected = reference.loc[period]
        error = max(abs(source["ff5_alpha_annual"] - expected.ff5_alpha_annual),
                    abs(source["ff5_alpha_hac_t"] - expected.ff5_alpha_hac_t))
        if not error < 1e-10:
            raise AssertionError(f"FF5 alpha does not reproduce the report for {period}: {error}")
        checks[f"report_ff5_alpha_{period}"] = dict(alpha=source["ff5_alpha_annual"], t=source["ff5_alpha_hac_t"],
                                                     max_abs_difference=error)
    metrics = pd.DataFrame(rows)
    for spec in SPECS:
        metrics[f"{spec}_alpha_holm_p"] = np.nan
        for period, part in metrics[metrics.role.eq("test")].groupby("period"):
            metrics.loc[part.index, f"{spec}_alpha_holm_p"] = holm_adjust(part[f"{spec}_alpha_hac_p"].to_numpy())
    out = RESULTS / EXP
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    changes = _git("status", "--porcelain", "--", "src", "data", "scripts", "tests", "notebooks", "paper",
                   "results/reference", "results/validation")
    manifest = dict(
        experiment=EXP, title="FF5 알파에 BAB 추가 (⑨)", run_at=datetime.now().isoformat(timespec="seconds"),
        command=" ".join(["python", *sys.argv]), git_commit=_git("rev-parse", "HEAD"),
        original_files_modified=bool(changes), python=platform.python_version(),
        returns_source="results/reference/monthly_returns.csv (total_observed_drift10)",
        returns_sha256=_sha(ROOT / "results/reference/monthly_returns.csv"),
        bab_source="AQR public US BAB series; see extensions/external_data", checks=checks,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(out, metrics, manifest)
    rebuild_ledger()
    if changes:
        raise RuntimeError("Original project files changed:\n" + changes)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)
    cols = ["variant", "period", "ff5_alpha_annual", "ff5_alpha_hac_t", "ff5_bab_alpha_annual", "ff5_bab_alpha_hac_t",
            "ff5_bab_alpha_hac_p", "ff5_bab_alpha_holm_p", "ff5_bab_beta_BAB", "ff5_bab_t_BAB", "r2_change"]
    print(metrics[cols].round(4).to_string(index=False))
    print(checks)


def write_summary(out, metrics, manifest):
    def line(r, label):
        holm = "-" if pd.isna(r.ff5_bab_alpha_holm_p) else f"{r.ff5_bab_alpha_holm_p:.3f}"
        return (f"| {label} | {100 * r.ff5_alpha_annual:.2f} | {r.ff5_alpha_hac_t:.2f} | "
                f"{100 * r.ff5_bab_alpha_annual:.2f} | {r.ff5_bab_alpha_hac_t:.2f} | {r.ff5_bab_alpha_hac_p:.3f} | "
                f"{holm} | {r.ff5_bab_beta_BAB:+.3f} ({r.ff5_bab_t_BAB:.2f}) | {100 * r.r2_change:+.1f} |")

    head = ("| {} | FF5 α(%) | t | FF5+BAB α(%) | t | p | Holm p | BAB 베타 (t) | R² 변화(pp) |\n"
            "|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    source = metrics[metrics.variant.eq(SOURCE)].set_index("period")
    full = metrics[metrics.period.eq("full")].set_index("variant")
    order = [*MAIN, *sorted(set(full.index) - set(MAIN))]
    lines = [
        f"## {EXP}: {manifest['title']}", "",
        "보고서 19개 전략의 비용 차감 후 월별 초과수익률을 FF5와 FF5+BAB에 각각 회귀했습니다. HAC(4) Bartlett, "
        "연율 알파는 월 절편의 12배로, 보고서와 같은 방식입니다. BAB는 AQR 공개 미국 월간 시계열입니다. "
        "포트폴리오를 새로 만들지 않고 보고서의 수익률을 그대로 썼습니다.", "",
        f"실행 시각: {manifest['run_at']}", "",
        "### 시가총액 가중 (발표 자료 슬라이드 22·56의 회귀)", "",
        head.format("기간"), *[line(source.loc[p], p) for p in PERIODS], "",
        "### 전체 기간, 19개 전략", "",
        "Holm p는 시가총액 가중을 뺀 18개 전략 안에서 조정했습니다.", "",
        head.format("전략"), *[line(full.loc[m], MAIN.get(m, f"`{m}`")) for m in order], "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
