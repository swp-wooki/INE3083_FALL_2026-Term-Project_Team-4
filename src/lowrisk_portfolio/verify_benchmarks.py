"""Reconcile the three market references without rerunning portfolio optimization."""

from .paths import OUTPUT, ROOT

P = OUTPUT
OUT = OUTPUT
import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

MARKETS = ["SP500", "CRSP_market", "FF_market"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics_from_returns(values, rf):
    values, rf = list(values), list(rf)
    excess = [r - f for r, f in zip(values, rf)]
    wealth = peak = 1.0
    worst = 0.0
    for r in values:
        wealth *= 1 + r
        peak = max(peak, wealth)
        worst = min(worst, wealth / peak - 1)
    mean = statistics.mean(values)
    return dict(
        months=len(values),
        cumulative_return=wealth - 1,
        cagr=wealth ** (12 / len(values)) - 1,
        annual_mean_return=12 * mean,
        report_annual_return=(1 + mean) ** 12 - 1,
        annual_volatility=statistics.stdev(values) * math.sqrt(12),
        sharpe=statistics.mean(excess) / statistics.stdev(excess) * math.sqrt(12),
        max_drawdown=worst,
        positive_month_fraction=sum(r > 0 for r in values) / len(values),
    )


def benchmark_rows(article, label):
    """Read benchmark rows across every tabular block within the labeled float."""
    block = article[article.index(r"\label{" + label + "}") :]
    end = re.search(r"\\end\{table\*?\}", block)
    if end is None:
        raise ValueError(f"No table end found for {label}")
    return [line for line in block[: end.start()].splitlines() if r"\hyperref[alloc:SP500]" in line]


def verify(previous=None):
    source = ROOT / "data/raw/market_benchmarks.csv"
    raw = pd.read_csv(source).set_index("month")
    b = pd.read_csv(P / "benchmark_returns.csv").set_index("month")
    bm = pd.read_csv(P / "benchmark_metrics.csv")
    g = pd.read_csv(P / "regime_metrics.csv")
    coverage = pd.read_csv(P / "regime_coverage.csv")
    returns = pd.read_csv(P / "monthly_returns.csv")
    wide = returns.query('scenario == "total_observed_drift10"').pivot(
        index="month", columns="model", values="net_return"
    )
    figure = json.loads((P / "figures/figure_manifest.json").read_text())
    checks = []

    def check(name, error, tolerance=0.0):
        checks.append(
            dict(
                check=name,
                error=float(error),
                tolerance=tolerance,
                passed=bool(math.isfinite(error) and error <= tolerance),
            )
        )

    check(
        "Complete unique 215-month benchmark panel",
        int(
            not b.index.is_unique
            or list(b.index) != list(pd.period_range("2003-02", "2020-12", freq="M").astype(str))
            or not np.isfinite(b.to_numpy()).all()
        ),
    )
    check(
        "S&P is the supplied sprtrn price series",
        np.max(np.abs(b.SP500 - raw.loc[b.index, "sp500_return"])),
    )
    check(
        "CRSP remains the supplied total-return series",
        np.max(np.abs(b.CRSP_market - raw.loc[b.index, "vw_market_return"])),
        1e-15,
    )
    periods = {
        "full": ("2003-02", "2020-12"),
        "2003_2011": ("2003-02", "2011-12"),
        "2012_2020": ("2012-01", "2020-12"),
        "financial_crisis": ("2007-07", "2009-03"),
        "covid": ("2020-02", "2020-06"),
    }
    for model in MARKETS:
        for period, (start, end) in periods.items():
            block = b.loc[start:end]
            row = bm[(bm.model == model) & (bm.period == period)]
            assert len(row) == 1
            expected = statistics_from_returns(block[model], block.RF)
            check(
                model + " metrics: " + period,
                max(abs(float(row.iloc[0][key]) - value) for key, value in expected.items()),
                1e-10,
            )
        for regime in coverage.itertuples():
            block = b.loc[regime.observed_start : regime.observed_end]
            row = g[(g.model == model) & (g.regime == regime.regime) & (g.scenario == "benchmark")]
            assert len(row) == 1
            expected = statistics_from_returns(block[model], block.RF)
            check(
                model + " regime: " + regime.regime,
                max(abs(float(row.iloc[0][key]) - value) for key, value in expected.items()),
                1e-10,
            )
    check("All 27 market-regime rows present", abs(len(g[g.scenario == "benchmark"]) - 27))
    check(
        "All three markets in both path figures",
        int(not set(MARKETS).issubset(figure["figure_series"])),
    )

    all_returns = wide.join(b[MARKETS])
    heat = pd.read_csv(P / "figures/regime_return_differences_pp.csv").set_index("model")
    heat_error = 0.0
    for regime in coverage.itertuples():
        block = all_returns.loc[regime.observed_start : regime.observed_end]
        base = math.prod(1 + v for v in block.source_weight)
        for model in heat.index:
            expected = 100 * (math.prod(1 + v for v in block[model]) - base)
            heat_error = max(heat_error, abs(heat.loc[model, regime.regime] - expected))
    check(
        "Heatmap includes three markets and 18 alternatives",
        int(heat.shape != (21, 9) or not set(MARKETS).issubset(heat.index)),
    )
    check("Every heatmap cell independently reconciled", heat_error, 1e-8)

    pattern = r"\\addplot\[[^\n]*?\] coordinates \{([^\n]*)\};"
    full_code = (P / "figures/wealth_drawdown.tex").read_text()
    regime_code = (P / "figures/regime_paths.tex").read_text()
    full_paths = re.findall(pattern, full_code)
    regime_paths = re.findall(pattern, regime_code)
    series = figure["figure_series"]
    assert len(full_paths) == 2 * len(series) and len(regime_paths) == len(coverage) * len(series)

    def path_error(text, expected):
        pairs = re.findall(r"\(([^,]+),([^\)]+)\)", text)
        assert len(pairs) == len(expected)
        assert [int(float(x)) for x, y in pairs] == list(range(len(expected)))
        return max(abs(float(y) - v) for (_, y), v in zip(pairs, expected))

    def paths(values):
        wealth = [1.0]
        for r in values:
            wealth.append(wealth[-1] * (1 + r))
        peak = 1.0
        dd = []
        for value in wealth:
            peak = max(peak, value)
            dd.append(100 * (value / peak - 1))
        return wealth, dd

    error = 0.0
    for i, model in enumerate(series):
        wealth, dd = paths(all_returns[model])
        error = max(
            error,
            path_error(full_paths[i], wealth),
            path_error(full_paths[len(series) + i], dd),
        )
    for j, regime in enumerate(coverage.itertuples()):
        for i, model in enumerate(series):
            wealth, _ = paths(all_returns.loc[regime.observed_start : regime.observed_end, model])
            error = max(error, path_error(regime_paths[j * len(series) + i], wealth))
    check("All full-period and regime plot coordinates reconciled", error, 5.1e-9)

    article = (P / "paper/main.tex").read_text()
    for label, expected_rows in [("tab:main", 1), ("tab:period", 9)]:
        rows = benchmark_rows(article, label)
        check(
            label + ": S&P rows highlighted",
            int(
                len(rows) != expected_rows
                or any(r"\textbf{" not in line or r"\underline{" not in line for line in rows)
            ),
        )

    if previous:
        old_b = pd.read_csv(previous / "benchmark_returns.csv").set_index("month")
        check(
            "Existing market and RF observations preserved",
            float(np.max(np.abs(old_b.to_numpy() - b[old_b.columns].to_numpy()))),
            1e-15,
        )
        for name, sha in json.loads((previous / "preserved_hashes.json").read_text()).items():
            check(name + " unchanged", int(digest(P / name) != sha))

    used = [
        "benchmark_returns.csv",
        "benchmark_metrics.csv",
        "regime_metrics.csv",
    ]
    result = dict(
        complete=all(c["passed"] for c in checks),
        checks=checks,
        source_sha256=digest(source),
        source_column="sp500_return (original sprtrn)",
        return_definition="S&P 500 price return; excludes dividends",
        primary_cost="20 bp round trip for strategies; no index cost deducted",
        artifacts_sha256={name: digest(P / name) for name in used},
        code_sha256={
            name: digest(Path(__file__).parent / name)
            for name in ["report.py", "figures.py", Path(__file__).name]
        },
    )
    (P / "market_benchmark_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["complete"], [c for c in checks if not c["passed"]]
    print(f"{len(checks)} benchmark checks passed; portfolio optimization not rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path)
    verify(parser.parse_args().previous)
