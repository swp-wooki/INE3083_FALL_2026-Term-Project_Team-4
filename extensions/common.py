"""Shared inputs, accounting, inference, and result ledger for extension experiments.

The report package under src/ is imported read-only. Every output of this folder goes
to results/extensions/<experiment>/ and the cumulative files beside those folders.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/extensions"
# The package resolves this path on import; keep it away from results/reproduced.
os.environ["LOWRISK_OUTPUT_DIR"] = str(RESULTS / "_package_output_unused")

import numpy as np
import pandas as pd

from lowrisk_portfolio.config import Config
from lowrisk_portfolio.data import load_panel
from lowrisk_portfolio.experiment import CAP, PRIMARY, PRIMARY_COST_BPS, evaluate
from lowrisk_portfolio.metrics import hac_regression, performance
from lowrisk_portfolio.models import module, validate_weights
from lowrisk_portfolio.paths import DATA
from lowrisk_portfolio.selection import holm_adjust, load_selection

N = 30
SOURCE = "source_weight"
SCORES = ["factor_1", "factor_2", "factor_3"]  # Low beta, low volatility, negative coskewness.
BOOT_REPS, BOOT_BLOCK, BOOT_SEED = 2000, 6, 20261002
COVARIANCE = {"sample": "cov_sample", "cc": "cov_constant_corr", "lw": "cov_lw_identity"}
# Allocation names used by the supplied results, for the reproduction checks.
REFERENCE_MODEL = {"sample": "gmvp_sample", "cc": "gmvp_constant_corr", "lw": "gmvp_lw_identity"}


@dataclass
class Inputs:
    selection: pd.DataFrame
    panel: object
    months: pd.PeriodIndex
    ids: pd.Index
    returns: pd.DataFrame
    realized: pd.DataFrame
    ended: pd.DataFrame
    names: dict
    scores: dict
    source: dict
    periods: dict


def load_inputs():
    path = DATA / "selections/defensive_selection.csv"
    s = load_selection(path, N)
    panel = load_panel(Config())
    months = pd.PeriodIndex(sorted(s.holding_month.unique()), freq="M")
    ids = pd.Index(sorted(s.PERMNO.unique()))
    returns = panel["total_return"]
    ended = panel["delisted"].loc[months, ids]
    realized = pd.DataFrame(0.0, index=months, columns=ids)
    names, scores, source = {}, {}, {}
    for m, g in s.groupby("holding_month"):
        g = g.set_index("PERMNO")
        names[m] = pd.Index(g.index)
        held = returns.loc[m, names[m]]
        if not np.isfinite(held.to_numpy()).all():
            raise ValueError(f"Unobserved selected return {m}")
        realized.loc[m, names[m]] = held
        scores[m] = g[SCORES].to_numpy()
        # The same six-decimal projection as the Market-cap weight in experiment.py.
        projected = module("factor_cov")._proj(g.weight.to_numpy(), CAP)
        source[m] = validate_weights(projected, names[m], CAP)
    periods = {"full": (str(months[0]), str(months[-1])), "2003_2011": ("2003-02", "2011-12"),
               "2012_2020": ("2012-01", "2020-12")}
    coverage = pd.read_csv(ROOT / "results/reference/regime_coverage.csv")
    for row in coverage.itertuples():
        periods[row.regime] = (row.observed_start, row.observed_end)
    return Inputs(s, panel, months, ids, returns, realized, ended, names, scores, source, periods)


def history_run(inp, m):
    """Longest contiguous block of observed returns for all selected names, ending at t."""
    observed = inp.returns.loc[: m - 1, inp.names[m]].notna().all(axis=1).to_numpy()[::-1]
    return len(observed) if observed.all() else int(np.argmin(observed))


def history(inp, m, window):
    """Balanced estimation window of at most `window` months; stocks are never substituted."""
    used = min(window, history_run(inp, m))
    h = inp.returns.loc[m - used : m - 1, inp.names[m]]
    if len(h) != used or used < 24 or not np.isfinite(h.to_numpy()).all():
        raise ValueError(f"Incomplete estimation history {m}")
    return h


def covariance(kind, history_frame):
    return module(COVARIANCE[kind]).estimate(history_frame.to_numpy())


def frank_wolfe_gap(S, w, cap):
    """Bound on f(w)-f* for w'Sw over the capped simplex, relative to f(w)."""
    g = 2 * S @ w
    remaining, best = 1.0, 0.0
    for i in np.argsort(g):
        take = min(cap, remaining)
        best += take * g[i]
        remaining -= take
        if remaining <= 0:
            break
    return float(max(g @ w - best, 0) / (w @ S @ w))


def empty_targets(inp):
    return pd.DataFrame(0.0, index=inp.months, columns=inp.ids)


def source_targets(inp):
    targets = empty_targets(inp)
    for m in inp.months:
        targets.loc[m, inp.names[m]] = inp.source[m]
    return targets


def exposure_tilts(inp, targets):
    """Average weighted-score difference from Market-cap weight, in percentage points."""
    rows = []
    for m in inp.months:
        w = targets.loc[m, inp.names[m]].to_numpy()
        rows.append((w - inp.source[m].to_numpy()) @ inp.scores[m])
    return 100 * np.mean(rows, axis=0)


def _performance(net, inp, with_factors):
    f = inp.panel.factors
    if not with_factors:
        return performance(net, f.RF)
    stats = performance(net, f.RF, inp.panel.market, f)
    excess = net - f.RF.reindex(net.index)
    five = f.reindex(net.index)[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].to_numpy()
    beta, t, p = hac_regression(excess, five, lags=4)
    stats.update(ff5_alpha_annual=12 * beta[0], ff5_alpha_hac_t=t[0], ff5_alpha_hac_p=p[0])
    return stats


def compute(inp, variants):
    """variants: {name: {"targets": frame, "role": test|baseline, "anchor": name|None, ...}}."""
    series, metrics = {}, []
    for name, v in variants.items():
        x = evaluate(v["targets"], inp.realized, inp.ended, PRIMARY_COST_BPS)
        x.index = pd.PeriodIndex(x.month, freq="M")
        series[name] = x
        meta = {k: val for k, val in v.items() if k != "targets"}
        tilts = exposure_tilts(inp, v["targets"])
        for period, (a, b) in inp.periods.items():
            sub = x.loc[a:b]
            metrics.append(dict(
                variant=name, period=period, **meta,
                **_performance(sub.net_return, inp, period in ("full", "2003_2011", "2012_2020")),
                annual_gross_mean=12 * sub.gross_return.mean(),
                mean_turnover=sub.turnover.mean(),
                mean_effective_n=sub.effective_n.mean(),
                mean_max_weight=sub.max_weight.mean(),
                tilt_low_beta_pp=tilts[0], tilt_low_vol_pp=tilts[1], tilt_coskew_pp=tilts[2],
            ))
    return series, pd.DataFrame(metrics)


def inference(inp, variants, series):
    """Paired tests versus Market-cap weight and each variant's anchor, as in the final report."""
    net = pd.DataFrame({name: series[name].net_return for name in variants})
    tests = [name for name, v in variants.items() if v["role"] == "test"]
    rows = []
    for name in net.columns:
        if name == SOURCE:
            continue
        beta, t, p = hac_regression(net[name] - net[SOURCE], lags=4)
        rows.append(dict(variant=name, baseline=SOURCE, annualized_difference=12 * beta[0],
                         hac_t=t[0], hac_p=p[0]))
        anchor = variants[name].get("anchor")
        if anchor and anchor != name:
            beta, t, p = hac_regression(net[name] - net[anchor], lags=4)
            rows.append(dict(variant=name, baseline=anchor, annualized_difference=12 * beta[0],
                             hac_t=t[0], hac_p=p[0]))
    paired = pd.DataFrame(rows)
    paired["family"] = np.where(paired.baseline.eq(SOURCE), "vs_market_cap", "vs_anchor")
    paired["holm_p"] = np.nan
    for _, idx in paired[paired.variant.isin(tests)].groupby("family").groups.items():
        paired.loc[idx, "holm_p"] = holm_adjust(paired.loc[idx, "hac_p"].to_numpy())
    # Identical resampled months across all variants; same scheme and seed as report.py.
    excess = net.to_numpy() - inp.panel.factors.RF.reindex(net.index).to_numpy()[:, None]
    rng = np.random.default_rng(BOOT_SEED)
    T = len(net)
    sr = excess.mean(0) / excess.std(0, ddof=1) * np.sqrt(12)
    samples = []
    for _ in range(BOOT_REPS):
        starts = rng.integers(0, T, size=(T + BOOT_BLOCK - 1) // BOOT_BLOCK)
        e = excess[((starts[:, None] + np.arange(BOOT_BLOCK)) % T).ravel()[:T]]
        samples.append(e.mean(0) / e.std(0, ddof=1) * np.sqrt(12))
    samples = np.array(samples)
    delta, low, high = [], [], []
    for row in paired.itertuples():
        i, j = net.columns.get_loc(row.variant), net.columns.get_loc(row.baseline)
        lo, hi = np.quantile(samples[:, i] - samples[:, j], [0.025, 0.975])
        delta.append(sr[i] - sr[j]), low.append(lo), high.append(hi)
    paired["delta_sharpe"], paired["sharpe_ci_low"], paired["sharpe_ci_high"] = delta, low, high
    return paired


def check_reference(variants, series):
    """Settings identical to the final report must reproduce its monthly net returns."""
    reference = pd.read_csv(ROOT / "results/reference/monthly_returns.csv")
    reference = reference[reference.scenario.eq(PRIMARY)]
    checks = {}
    for name, v in variants.items():
        model = v.get("reference_model")
        if model:
            expected = reference[reference.model.eq(model)].set_index("month").net_return
            actual = series[name].set_index("month").net_return.reindex(expected.index)
            error = float((actual - expected).abs().max())
            if not error < 1e-7:
                raise AssertionError(f"{name} does not reproduce {model}: {error}")
            checks[name] = dict(reference_model=model, max_abs_monthly_difference=error)
    return checks


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pct(x, digits=2):
    return f"{100 * x:.{digits}f}"


def _summary_table(metrics, paired, variants):
    full = metrics[metrics.period.eq("full")].set_index("variant")
    market = paired[paired.family.eq("vs_market_cap")].set_index("variant")
    anchor = paired[paired.family.eq("vs_anchor")].set_index("variant")
    lines = [
        "| 설정 | CAGR | 변동성 | 샤프 | MDD | 월 회전율 | 유효종목수 | 저베타/저변동/코스큐 틸트(pp) "
        "| 시총가중 대비 평균차(pp) | Holm p | 샤프차 [95% 구간] | 기준 대비 샤프차 [95% 구간] |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in variants:
        r = full.loc[name]
        row = [f"`{name}`", _pct(r.cagr), _pct(r.annual_volatility), f"{r.sharpe:.3f}",
               _pct(r.max_drawdown), _pct(r.mean_turnover), f"{r.mean_effective_n:.1f}",
               f"{r.tilt_low_beta_pp:+.2f} / {r.tilt_low_vol_pp:+.2f} / {r.tilt_coskew_pp:+.2f}"]
        if name in market.index:
            q = market.loc[name]
            holm = "-" if pd.isna(q.holm_p) else f"{q.holm_p:.3f}"
            row += [f"{100 * q.annualized_difference:+.2f}", holm,
                    f"{q.delta_sharpe:+.3f} [{q.sharpe_ci_low:+.3f}, {q.sharpe_ci_high:+.3f}]"]
        else:
            row += ["-", "-", "-"]
        if name in anchor.index:
            q = anchor.loc[name]
            row.append(f"{q.delta_sharpe:+.3f} [{q.sharpe_ci_low:+.3f}, {q.sharpe_ci_high:+.3f}]")
        else:
            row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def publish(exp_id, title, design, inp, variants, checks=None, extra_files=None):
    """Write one experiment's files, then upsert it into the cumulative result files."""
    out = RESULTS / exp_id
    out.mkdir(parents=True, exist_ok=True)
    series, metrics = compute(inp, variants)
    paired = inference(inp, variants, series)
    checks = dict(checks or {}, reference_reproduction=check_reference(variants, series))
    metrics.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "paired_inference.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({k: x.net_return for k, x in series.items()}).rename_axis("month").to_csv(
        out / "monthly_net_returns.csv")
    positions = []
    for name, v in variants.items():
        long = v["targets"].stack().rename("weight").rename_axis(["month", "permno"]).reset_index()
        positions.append(long[long.weight.ne(0)].assign(variant=name))
    pd.concat(positions).to_csv(out / "holdings.csv.gz", index=False)
    for name, frame in (extra_files or {}).items():
        frame.to_csv(out / name, index=False, encoding="utf-8-sig")
    src_changes = _git("status", "--porcelain", "--", "src", "data", "scripts", "tests",
                       "notebooks", "paper", "results/reference", "results/validation")
    manifest = dict(
        experiment=exp_id, title=title, run_at=datetime.now().isoformat(timespec="seconds"),
        command=" ".join([Path(sys.executable).name, *sys.argv]), git_commit=_git("rev-parse", "HEAD"),
        original_files_modified=bool(src_changes), python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, primary_one_way_cost_bps=PRIMARY_COST_BPS,
        selection_sha256=_sha(DATA / "selections/defensive_selection.csv"),
        code_sha256={p.name: _sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        checks=checks,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [f"## {exp_id}: {title}", "", design, "", f"실행 시각: {manifest['run_at']}", "",
               "전체 기간(2003-02~2020-12, 215개월), 편도 10bp 비용 차감 후 수익률 기준입니다. "
               "Holm p는 `test` 설정끼리만 조정했고 `baseline`은 `-`로 표시합니다.", "",
               _summary_table(metrics, paired, variants), ""]
    (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    rebuild_ledger()
    if src_changes:
        raise RuntimeError("Original project files changed:\n" + src_changes)
    return series, metrics, paired


def rebuild_ledger():
    """Regenerate the cumulative CSV and Markdown from every experiment folder."""
    folders = sorted(p for p in RESULTS.iterdir() if (p / "manifest.json").exists())
    frames, index, sections = [], [], []
    for folder in folders:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        frames.append(pd.read_csv(folder / "metrics.csv", encoding="utf-8-sig").assign(experiment=folder.name))
        index.append(f"| `{folder.name}` | {manifest['title']} | {manifest['run_at']} |")
        sections.append((folder / "summary.md").read_text(encoding="utf-8"))
        if (folder / "notes.md").exists():
            sections.append((folder / "notes.md").read_text(encoding="utf-8"))
    combined = pd.concat(frames, ignore_index=True)
    first = ["experiment", "variant", "period"]
    combined = combined[first + [c for c in combined.columns if c not in first]]
    combined.to_csv(RESULTS / "all_results.csv", index=False, encoding="utf-8-sig")
    header = [
        "# 확장 실험 결과 모음", "",
        "실험을 실행할 때마다 자동으로 다시 만들어지는 파일입니다. 직접 수정하면 다음 실행 때 사라지니, "
        "해석은 각 실험 폴더의 `notes.md`에 적습니다. 전체 수치(기간·국면별)는 `all_results.csv`에 있습니다.", "",
        "공통 조건: 팀의 월별 30종목 선택 고정, 2003-02~2020-12, 편도 10bp, 공매도 금지, 월별 리밸런싱. "
        "회계와 통계 방법은 최종 보고서 코드를 그대로 사용합니다.", "",
        "| 실험 | 내용 | 마지막 실행 |", "|:--|:--|:--|", *index, "",
    ]
    shared = RESULTS / "settings_and_cautions.md"
    if shared.exists():
        header += [shared.read_text(encoding="utf-8"), "# 실험별 결과", ""]
    (RESULTS / "RESULTS.md").write_text("\n".join(header + sections), encoding="utf-8")
