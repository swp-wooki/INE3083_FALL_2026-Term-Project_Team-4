"""Regression coverage for the report's two-column regime table."""

import unittest

from lowrisk_portfolio.verify_benchmarks import benchmark_rows


class ReportVerificationTests(unittest.TestCase):
    def test_rows_after_first_tabular_are_included(self):
        source = r"""\begin{table*}\label{tab:period}
\begin{tabular}{ll}
\textbf{\hyperref[alloc:SP500]{S\&P 500}} & \underline{1.0} \\
\bottomrule\end{tabular}
\begin{tabular}{ll}
\textbf{\hyperref[alloc:SP500]{S\&P 500}} & \underline{2.0} \\
\bottomrule\end{tabular}\end{table*}
\hyperref[alloc:SP500]{outside the table}
"""
        rows = benchmark_rows(source, "tab:period")
        self.assertEqual(len(rows), 2)
        self.assertIn("2.0", rows[1])

    def test_unclosed_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No table end"):
            benchmark_rows(r"\label{tab:main}\bottomrule", "tab:main")


if __name__ == "__main__":
    unittest.main()
