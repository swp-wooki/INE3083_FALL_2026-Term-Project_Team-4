"""Regression checks for the reviewed manuscript's tables and figures."""

import unittest

from lowrisk_portfolio.figures import embed_figures
from lowrisk_portfolio.paths import PAPER
from lowrisk_portfolio.report import replace_rows, validate_selection_section
from lowrisk_portfolio.verify_benchmarks import benchmark_emphasis_ok, benchmark_rows


class ReportVerificationTests(unittest.TestCase):
    def test_reviewed_manuscript_has_balanced_audit_and_current_benchmark(self):
        manuscript = (PAPER / "main.tex").read_text()
        validate_selection_section(manuscript)
        for label, count in [("tab:main", 1), ("tab:period", 9)]:
            with self.subTest(label=label):
                self.assertTrue(benchmark_emphasis_ok(manuscript, label, count))

    def test_regime_rows_span_both_tabular_blocks_and_stop_at_float_end(self):
        panel = r"""\begin{tabular}{lr}
\textbf{\hyperref[alloc:CRSP_market]{CRSP market}} & \underline{1.0} \\
\hyperref[alloc:FF_market]{FF market} & 1.1 \\
\hyperref[alloc:SP500]{S\&P 500} & 1.2 \\
\bottomrule\end{tabular}
"""
        source = (
            r"\begin{table*}\label{tab:period}"
            + "\n"
            + panel * 2
            + r"\end{table*}"
            + "\n"
            + r"\hyperref[alloc:CRSP_market]{outside the table}"
        )
        self.assertEqual(len(benchmark_rows(source, "tab:period")), 2)
        self.assertTrue(benchmark_emphasis_ok(source, "tab:period", 2))
        wrong = source.replace(
            r"\hyperref[alloc:SP500]{S\&P 500}", r"\textbf{\hyperref[alloc:SP500]{S\&P 500}}"
        )
        self.assertFalse(benchmark_emphasis_ok(wrong, "tab:period", 2))
        self.assertFalse(
            benchmark_emphasis_ok(source.replace(r"\underline{1.0}", "1.0"), "tab:period", 2)
        )

    def test_unclosed_table_cannot_consume_the_next_float(self):
        with self.assertRaisesRegex(ValueError, "No table end"):
            benchmark_rows(r"\label{tab:main}\bottomrule\begin{table}\end{table}", "tab:main")

    def test_shared_tabular_panel_is_replaced_without_overwriting_neighbor(self):
        source = r"""\begin{table}\label{tab:archive}
\textit{Panel A.}
Head A \\\midrule
Old A \\\midrule
\multicolumn{7}{l}{\textit{Panel B.}}
Head B \\\midrule
Keep B \\\bottomrule
\end{table}
Following narrative."""
        actual = replace_rows(source, "tab:archive", r"New A \\", panel="A")
        self.assertIn(r"New A \\", actual)
        self.assertNotIn("Old A", actual)
        self.assertEqual(source.split(r"\multicolumn")[1], actual.split(r"\multicolumn")[1])
        self.assertEqual(replace_rows(actual, "tab:archive", r"New A \\", panel="A"), actual)

    def test_selection_marker_cannot_swallow_conclusion(self):
        source = "% BEGIN SELECTION RISK AUDIT\n" + r"\label{tab:selectionrisk}" + "\n"
        with self.assertRaisesRegex(ValueError, "matched BEGIN/END"):
            validate_selection_section(source)
        end = "% END SELECTION RISK AUDIT\n"
        validate_selection_section(source + end + r"\section{Conclusion}")
        with self.assertRaisesRegex(ValueError, "following section"):
            validate_selection_section(source + r"\section{Conclusion}" + "\n" + end)

    def test_figure_refresh_keeps_reviewed_placement_and_surrounding_prose(self):
        def figure(label, content):
            return r"\begin{figure*}" + content + r"\label{" + label + r"}\end{figure*}"

        a, b = "fig:regime_paths", "fig:regime_heatmap"
        source = figure(a, "old paths") + "\n\nReviewed passage.\n\n" + figure(b, "old heatmap")
        output = embed_figures(source, {a: figure(a, "new paths"), b: figure(b, "new heatmap")})
        self.assertLess(output.index("new paths"), output.index("Reviewed passage."))
        self.assertLess(output.index("Reviewed passage."), output.index("new heatmap"))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            embed_figures(source + "\n" + figure(a, "duplicate"), {a: figure(a, "new paths")})


if __name__ == "__main__":
    unittest.main()
