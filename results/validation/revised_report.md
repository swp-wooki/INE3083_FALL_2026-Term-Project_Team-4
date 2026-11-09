# Reproduction record for the reviewed manuscript

The comparison target before this repair is [the reviewed manuscript at 5753b19](https://github.com/swp-wooki/INE3083_FALL_2026-Term-Project_Team-4/blob/5753b19436d4453c332417c73d8889f53a3e4962/paper/main.tex). The original allocation return series and the reconstructed selection return series are separate inputs to inference. The older validation records at this directory's top level describe the original allocation release; they do not establish reproduction of the November revision.

| Result | Calculation | Inspect under `results/reproduced/` |
|:--|:--|:--|
| Table I, Panels A and B | Rerank the eligible universe and rebuild all seven selections under the same rules; compare B+V+C to B+V using paired months | `revisions/signal_metrics.csv`, `signal_increment.csv`, `signal_removals.csv` under `revisions/` |
| §III-A downside beta | Regress strategy excess returns on FF excess-market returns in negative excess-market months; rank tail months by FF total return | `revisions/downside_risk.csv` |
| Table III, Panel B | Use the original net allocation returns for the CRSP/FF market differences and FF5/FF5+BAB regressions | `market_inference.csv`, `extensions/ext09_alpha_bab/metrics.csv` |
| Table IX | Add each fourth signal under the same selection constraints; form and rebalance two-sleeve blends from the net sleeve returns | `revisions/fourth_signals_blends.csv`, `revisions/bootstrap_precision.csv` |
| Appendix A | Estimate FF5 break/Wald tests and rolling exposures on original returns. Independently build low-volatility and coskewness long–short controls for reconstructed selections | `revisions/temporal_interactions.csv`, `temporal_rolling.csv`, `temporal_regressions.csv`, `temporal_factor_means.csv` under `revisions/` |
| Appendix C, Tables XI–XII | Solve **49 additional** specifications; keep six baselines and one auxiliary EWMA diagnostic separately identified | `revisions/allocation_metrics.csv`, `allocation_risk_calibration.csv`, `allocation_summary.json` under `revisions/` |

**What matches the previous manuscript.** All seven Table I.A rows and all eight Table IX rows match at the displayed precision for the performance point estimates. Table I.B's full-period annual mean difference is 4.12 pp (HAC t = 3.18), its Sharpe gain is 0.281, and its spanning alpha is 4.04% (t = 4.27). Downside beta is 0.66 versus 0.79. Every displayed Sharpe value in Tables XI–XII matches (53 numeric cells across 19 rows). The 30- and 50-stock export scripts independently regenerate both archived UTF-8/CRLF files with identical SHA-256 digests.

**What was reconstructed rather than recovered.** The earlier selection bootstrap random stream and every supplementary estimator detail were not archived. The following differences are disclosed; numeric agreement with newly committed reference CSVs is a regression check, not evidence that the earlier numbers were identical.

| Quantity | Earlier manuscript | Reproducible calculation |
|:--|:--|:--|
| Table I.B full-sample Sharpe-gain interval | [0.051, 0.562] | [0.054, 0.553]; same point estimate |
| Table IX profitability addition, upper 95% Sharpe endpoint | +0.003 | −0.006 at 2,000 draws (seed 20261109); +0.00224 at 100,000 draws (seed 91007). The sign near zero is sensitive to Monte Carlo sampling. The 52-week-low upper endpoint remains negative. |
| Later-period Fama–MacBeth coskewness slope, pp/month (HAC t) | 0.20 (0.97) | 0.18 (0.86) using untrimmed observed next-month returns and equal-weighted cross-sectional least squares |
| Annualized coskewness spread, early / late | 5.93% / 0.56% | 5.96% / 0.58% using gross, uncapped, value-weighted upper/lower 30% legs |
| Realized / predicted annual volatility | Review draft: 1.23–1.66 across “36 months or more” | 1.23–1.66 for the 32 fixed 36-month GMVPs; 1.22–1.66 across 42 when the ten longer-window settings are included. Realized TE: 2.14–2.62% under the 2% bound and 3.07–3.68% under the 3% bound. |

Run `python -m lowrisk_portfolio.cli` and `python scripts/compare_reference.py`. The latter checks 35 CSV tables, including the original BAB branch outputs and the reconstructed series, and all 12 numbered floats in the PDF source at manuscript precision. The benchmark checker independently checks plotted return coordinates and the primary CRSP emphasis in both the figures and manuscript.
