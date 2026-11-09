# External BAB factor experiments

Both experiments use the committed November 3, 2026 vintage of AQR's U.S. monthly Betting Against Beta factor. The [source record](external_data/aqr_bab_usa.source.json) gives the workbook URL, sheet, column and original workbook checksum. The [input manifest](../data/manifest.json) identifies the bytes of the extracted factor used at runtime. No new download is required.

| Experiment | Estimand | Output |
|:--|:--|:--|
| E | GMVP weights and returns using FF5 and FF5+BAB covariance estimates under matched windows and caps | `extE_factor_cov/`: allocation performance, paired tests, holdings and forecast diagnostics |
| ⑨ | FF5 and FF5+BAB regressions on the 19 original strategy return series, with Bartlett HAC(4) inference | `ext09_alpha_bab/metrics.csv`: alphas, loadings, t-statistics and adjusted p-values for three periods |

Run `python -m lowrisk_portfolio.cli` from the repository root to compute both from the report inputs. The pipeline passes the newly generated net returns to ⑨. Its estimates in Table III are distinct from regressions on reconstructed selections in Appendix A. The factor itself is long–short; the strategy returns are long-only and net of costs.

For a standalone rerun with the allocation environment installed:

```bash
python extensions/extE_factor_cov.py
python extensions/ext09_alpha_bab.py
```

The standalone regression uses the supplied reference returns by default. Set `LOWRISK_EXTENSION_INPUT_DIR` to use a directory containing regenerated `monthly_returns.csv` and `market_inference.csv`. Output goes to `results/reproduced/extensions/`, or to the `extensions/` subdirectory of `LOWRISK_OUTPUT_DIR` when set. The original branch's records remain in `results/extensions/` for comparison.
