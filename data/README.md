# Data inventory

All files are supplied project inputs. Descriptive filenames identify their role; source provenance remains documented below. Original contents and column names are preserved. Large plain CSVs are stored with lossless gzip compression; their decompressed bytes match the originals exactly. SHA-256 values and sizes for both stored files and original payloads are recorded in `manifest.json`.

| File | Source and content | Role |
|:--|:--|:--|
| `raw/security_monthly.csv.gz` | Original CRSP monthly CIZ extract; returns, capitalization, identifiers, and security end dates | Required; restores observed returns omitted from the filtered extract |
| `raw/security_prices.csv.gz` | Course-provided stock price panel | Required; establishes security/month coverage and diagnostic price returns |
| `processed/security_monthly_clean.csv.gz` | Supplied filtered CRSP monthly extract | Required; observed returns and metadata, including original metadata missingness |
| `raw/market_benchmarks.csv` | Supplied CRSP broad-market total returns and S&P 500 price returns | Required market comparisons |
| `raw/factor_returns_monthly.csv` | Fama–French five factors and risk-free returns; supplied July 2026 vintage | Required factor covariance and inference |
| `selections/defensive_selection.csv` | Supplied low-risk/coskewness selection; 30 stocks per holding month | Primary experiment; 6,450 rows |
| `selections/return_oriented_selection.csv` | Supplied momentum/profitability/issuance selection; 50 stocks per holding month | Report's selection comparison |
| `supplementary/company_fundamentals.csv.gz` | Original Compustat fundamentals extract | Included source archive; not consumed by the final experiment |

## Units and alignment

- Security identifiers are retained as `PERMNO` or `permno`, depending on the supplied schema.
- `MthRet` and saved strategy returns are decimal monthly returns. `MthCap` retains the source unit of thousands of dollars.
- The factor source reports percentages; the loader divides by 100 exactly once.
- Selection weights and factor scores are decimal fractions. `holding_month` must equal `formation_month + 1`.
- The primary experiment uses observed total returns. A price-return diagnostic in the audit table is not used as a fallback for missing strategy returns.
- The original raw and prepared files are both included. The source procedure that produced the prepared extract and upstream daily selection inputs was not supplied; these prepared files are explicit reproduction inputs.

## Integrity and retrieval

Four large files use Git LFS. After cloning, run `git lfs pull`, then:

```bash
python scripts/verify_data.py
```

Use `--required-only` to skip hashing the supplementary fundamentals archive. The complete check validates every file, including the unused original archive.

The loaders read `.csv.gz` files directly. To recover a plain original, use `gzip -dk data/raw/security_monthly.csv.gz` (or decompress with Python’s `gzip` module). This is optional for every documented execution path.
