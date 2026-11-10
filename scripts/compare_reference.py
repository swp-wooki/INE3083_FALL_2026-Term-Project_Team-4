"""Compare reproduced tables with the supplied final-report numerical reference."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "data_audit.csv": ["holding_month", "permno"],
    "monthly_returns.csv": ["scenario", "model", "month"],
    "holdings.csv.gz": ["model", "holding_month", "permno"],
    "model_metrics.csv": ["scenario", "model", "period"],
    "paired_comparisons.csv": ["scenario", "period", "baseline", "model"],
    "benchmark_returns.csv": ["month"],
    "benchmark_metrics.csv": ["model", "period"],
    "regime_metrics.csv": ["regime", "scenario", "model"],
    "sharpe_bootstrap.csv": ["model", "baseline"],
    "mechanism_summary.csv": ["model"],
    "selection_comparison/metrics.csv": ["selection", "model", "period"],
    "selection_comparison/paired_inference.csv": ["model"],
    "revisions/allocation_metrics.csv": ['variant', 'period'],
    "revisions/allocation_monthly_net_returns.csv": ['month'],
    "revisions/allocation_paired_inference.csv": ['variant', 'baseline', 'family'],
    "revisions/allocation_risk_calibration.csv": ['variant'],
    "revisions/allocation_score_matching.csv": ['variant', 'baseline', 'metric'],
    "revisions/bootstrap_precision.csv": ['model'],
    "revisions/capture_windows.csv": ['period', 'strategy'],
    "revisions/crisis_contribution.csv": ['market'],
    "revisions/downside_risk.csv": ['model'],
    "revisions/fourth_signals_blends.csv": ['model'],
    "revisions/joint_fama_macbeth.csv": ['period', 'signal'],
    "revisions/removal_contrasts.csv": ['period', 'removed_1', 'removed_2'],
    "revisions/residual_coskew.csv": ['model'],
    "revisions/residual_coskew_rank_correlation.csv": ['pair'],
    "revisions/selection_monthly.csv": ['month'],
    "revisions/selection_sensitivity.csv": ['setting'],
    "revisions/signal_increment.csv": ['period'],
    "revisions/signal_metrics.csv": ['period', 'model'],
    "revisions/signal_removals.csv": ['period', 'removed'],
    "revisions/temporal_factor_means.csv": ['period'],
    "revisions/temporal_interactions.csv": ['lags'],
    "revisions/temporal_regressions.csv": ['model', 'period'],
    "revisions/temporal_rolling.csv": ['window', 'ending'],
    "extensions/ext09_alpha_bab/metrics.csv": ["variant", "period"],
    "extensions/extE_factor_cov/forecast_vs_realized.csv": ["variant"],
}


def manuscript_tables(source):
    """Keep both panels/tabulars inside each numbered float; ignore harmless whitespace."""
    tables = {}
    for match in re.finditer(r"\\begin\{(table\*?)\}(.*?)\\end\{\1\}", source, re.S):
        body = match[2]
        label = re.search(r"\\label\{(tab:[^}]+)\}", body)
        if label is None:
            continue
        if label[1] in tables:
            raise ValueError(f"Duplicate table label: {label[1]}")
        tables[label[1]] = " ".join(body.split())
    return tables


def compare_manuscript(output):
    expected = manuscript_tables((ROOT / "paper/main.tex").read_text())
    actual = manuscript_tables((output / "paper/main.tex").read_text())
    if not expected or set(actual) != set(expected):
        raise AssertionError("Rebuilt report has missing or unexpected numbered tables")
    for label in expected:
        if actual[label] != expected[label]:
            raise AssertionError(
                f"Rebuilt {label} differs from the committed manuscript at displayed precision or formatting"
            )
    return sorted(actual)


def compare(output, atol=1e-8, rtol=1e-7):
    checks = []
    for name, keys in TABLES.items():
        actual = pd.read_csv(output / name).set_index(keys).sort_index()
        reference = ROOT / ("results" if name.startswith("extensions/") else "results/reference")
        expected = pd.read_csv(reference / name).set_index(keys, verify_integrity=True).sort_index()
        if not actual.index.equals(expected.index) or set(actual.columns) != set(expected.columns):
            raise AssertionError(f"Table structure changed: {name}")
        numeric = expected.select_dtypes(include="number").columns
        errors = []
        for column in numeric:
            np.testing.assert_allclose(
                actual[column],
                expected[column],
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                err_msg=f"{name}: {column}",
            )
            finite = np.isfinite(actual[column]) & np.isfinite(expected[column])
            if finite.any():
                errors.append(
                    float((actual.loc[finite, column] - expected.loc[finite, column]).abs().max())
                )
        for column in expected.columns.difference(numeric):
            pd.testing.assert_series_equal(actual[column], expected[column], check_names=False)
        checks.append(
            {
                "table": name,
                "rows": len(actual),
                "maximum_absolute_difference": max(errors, default=0.0),
            }
        )
    table_labels = compare_manuscript(output)
    report = {
        "manuscript_tables": table_labels,
        "status": "passed",
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "tables": checks,
    }
    (output / "reference_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"All {len(checks)} reference tables agree within atol={atol:g}, rtol={rtol:g}.")
    print(f"All {len(table_labels)} numbered manuscript tables match the committed report.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    compare(parser.parse_args().output.resolve())
