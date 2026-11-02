"""Compare reproduced tables with the supplied final-report numerical reference."""

import argparse
import json
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
}


def compare(output, atol=1e-8, rtol=1e-7):
    checks = []
    for name, keys in TABLES.items():
        actual = pd.read_csv(output / name).set_index(keys).sort_index()
        expected = pd.read_csv(ROOT / "results/reference" / name).set_index(keys).sort_index()
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
    report = {
        "status": "passed",
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "tables": checks,
    }
    (output / "reference_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"All {len(checks)} reference tables agree within atol={atol:g}, rtol={rtol:g}.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    compare(parser.parse_args().output.resolve())
