"""Regression checks for the two distinct calibration populations and table verification."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lowrisk_portfolio.allocation_sensitivity import (
    Setting,
    _risk_ranges,
    render_calibration,
    verify_table_rows,
)


class AllocationSensitivityTests(unittest.TestCase):
    def test_longer_window_and_non_gmvp_are_not_misreported_as_fixed_window(self):
        specs = [
            Setting("short", "lw", 12, 0.1, "gmvp", "short_window"),
            Setting("fixed", "cc", 36, 0.1, "te2", "tracking_error"),
            Setting("longer", "sample", 60, 0.1, "gmvp", "window_cap"),
            Setting("erc", "sample", 36, 1.0, "erc_nocap", "risk_budgeting"),
        ]
        risk = pd.DataFrame(
            {"realized_over_forecast": [0.8, 1.3, 1.2, 1.1]}, index=[s.name for s in specs]
        )
        result = _risk_ranges(risk, specs)
        self.assertEqual(result["fixed_36_month_gmvp"]["settings"], ["fixed"])
        self.assertEqual(result["36_or_longer_window_gmvp"]["minimum_setting"], "longer")

    def test_calibration_prose_reads_fresh_values_and_preserves_neighboring_text(self):
        text = (
            "Context. Across the 32 fixed 36-month GMVP specifications, old values. "
            "The numerator is the annualized standard deviation of net monthly returns; "
            "the denominator uses forecasts. Realized tracking error ranges from old values. "
            "Moving closer to Capped VW changes exposures."
        )
        summary = {
            "risk_calibration": {
                "fixed_36_month_gmvp": {"count": 3, "minimum": 1.111, "maximum": 1.555},
                "36_or_longer_window_gmvp": {"count": 5, "minimum": 1.083, "maximum": 1.555},
            },
            "realized_te_limit_2pct": [0.022, 0.024],
            "realized_te_limit_3pct": [0.031, 0.039],
        }
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "allocation_summary.json").write_text(json.dumps(summary))
            rendered = render_calibration(text, temp)
            self.assertEqual(rendered, render_calibration(rendered, temp))
            self.assertIn("Across the 3 fixed", rendered)
            self.assertIn("across all 5 GMVP specifications", rendered)
            self.assertIn(r"from 2.20\% to 2.40\%", rendered)
            self.assertTrue(rendered.endswith("Moving closer to Capped VW changes exposures."))

    def test_published_cell_change_fails_even_when_row_count_is_unchanged(self):
        source = r"""\label{tab:check}
        Title & Value\\\midrule
        Sample & 1.234\\
        \bottomrule"""
        self.assertEqual(verify_table_rows(source, "check", r"Sample & 1.234\\"), 1)
        with self.assertRaisesRegex(ValueError, r"tab:check row 1"):
            verify_table_rows(source, "check", r"Sample & 1.243\\")


if __name__ == "__main__":
    unittest.main()
