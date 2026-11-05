# Stock-selection code

This folder holds the original code that generated the two fixed selection files in `data/selections/`. The allocation experiment in `src/lowrisk_portfolio/` starts from those files. The code here rebuilds them from the inputs.

| Script | Output | Rule |
|:--|:--|:--|
| `export_selection_b.py` | `defensive_selection.csv` | Equal-weighted percentile composite of low Frazzini–Pedersen beta (`low_beta_fp`), low 12-month volatility (`low_vol_12m`) and negative coskewness (`coskew`). Top 30, buffer 1.5 (holdings kept while ranked within 45), at most 9 stocks per SIC2 industry |
| `export_selection_momentum.py` | `return_oriented_selection.csv` | Equal-weighted percentile composite of 52-week-low momentum (`high_52w_low`), profitability loading (`rmw_beta`) and low net issuance (`net_issuance`). Top 50, buffer 1.5, no industry cap |

Both use the same universe each formation month: stocks at or above the NYSE 20th market-cap percentile, with SPACs and stocks delisted that month removed. Weights are proportional to formation-month market cap, capped at 2/k per stock, with any excess reassigned in proportion to market cap.

## Modules

| File | Contents |
|:--|:--|
| `selection_eval.py` | Data loading and merging, the universe filter, Top-k and decile evaluation, and the input and output paths |
| `factor_research_v2.py` | Factor research stages s0–s10, percentile ranks, composite scores and `topk_weights` (buffer, industry cap, weight cap) |
| `signals.py`, `signals_v2.py`, `signals_tech.py` | Signal definitions. `low_beta_fp` is the 12-month mean of daily beta (`beta_d`, clipped to [−3, 5]) shrunk as 0.6·β + 0.4 |
| `delisting.py` | Infers delisting reasons. Performance-related and unclear delistings get −50% in the delisting month; mergers and reclassifications keep the CIZ return |

## Inputs

| Path | Original name | Content |
|:--|:--|:--|
| `data/raw/security_prices.csv.gz` | `IFE_term_stock_data.csv` | Course-provided monthly price panel. It defines the universe and returns |
| `data/selection_inputs/crsp_monthly_ciz.csv.gz` | `llfzpekib12cqzql.csv` | WRDS CRSP monthly CIZ extract (1999-12 to 2020-12, 20 columns): market cap, shares, SIC, delisting-month returns |
| `data/selection_inputs/daily_signals.csv.gz` | `daily_signals.csv` | Monthly characteristics computed from CRSP daily returns: daily beta `beta_d`, `rvol_d`, `ivol_d`, `max_d`, `min_d`, `skew_d`, `illiq_d`, `dvol_m` |
| `data/selection_inputs/ff5_monthly.csv` | `data/ff5_monthly.csv` | Fama–French five factors and RF in decimals, from the vintage downloaded when the selections were made |

`crsp_monthly_ciz.csv.gz` is a separate WRDS query, not a subset of `data/raw/security_monthly.csv.gz`: 8,806 of its rows, including all of December 1999, are missing from that file. The factor file also differs from `data/raw/factor_returns_monthly.csv`, which is the July 2026 vintage. Use the files listed here to reproduce the selections exactly. The code that turned CRSP daily records into `daily_signals.csv` was not kept, so that file is an input.

## Run

From the repository root, after `git lfs pull`:

```bash
python -m pip install -r selection/requirements.txt
python selection/export_selection_b.py
python selection/export_selection_momentum.py
```

Outputs and the `MarketData` cache (`v2/marketdata_classified.pkl`) are written to `results/reproduced/selection/`. Set `SELECTION_OUTPUT_DIR` to write somewhere else. The first run builds the cache, which takes several minutes. Later runs reuse it.

For a fresh run with Python 3.13.5 and the versions in `selection/requirements.txt`, both outputs matched the committed files in `data/selections/` byte for byte (same SHA-256). These versions differ from the allocation package's pins. Use a separate environment.
