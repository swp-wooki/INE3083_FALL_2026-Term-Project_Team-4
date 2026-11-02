"""Load the observed-return inputs used by the final report, without imputation."""

import re
from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd


@dataclass
class Panel:
    fields: dict
    factors: pd.DataFrame
    market: pd.Series
    audit: dict

    def __getitem__(self, key):
        return self.fields[key]


def read_factors(path):
    """Official monthly FF percent CSV -> decimal monthly returns (skip annual section)."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = next(s for s in lines if "Mkt-RF" in s and "RF" in s)
    body = [s for s in lines if re.match(r"^\s*\d{6},", s)]
    f = pd.read_csv(StringIO("\n".join([header] + body)), index_col=0)
    f.index = pd.PeriodIndex(f.index.astype(str), freq="M")
    f.columns = f.columns.str.strip()
    f = f.apply(pd.to_numeric).replace([-99.99, -999], np.nan) / 100
    if f.index.has_duplicates:
        raise ValueError("Duplicate factor months")
    return f


def load_panel(config):
    config.validate()
    root = config.data_dir
    ife = pd.read_csv(root / "raw/security_prices.csv.gz", usecols=["date", "PERMNO", "ADJ_PRC"])
    ife["month"] = pd.to_datetime(ife.date).dt.to_period("M")
    if ife.duplicated(["PERMNO", "month"]).any():
        raise ValueError("Duplicate IFE security-month keys")
    ife = ife.sort_values(["month", "PERMNO"])
    months = pd.period_range(ife.month.min(), ife.month.max(), freq="M")
    ids = np.sort(ife.PERMNO.unique())

    def wide(df, col, key="PERMNO"):
        return df.pivot(index="month", columns=key, values=col).reindex(index=months, columns=ids)

    ife["ADJ_PRC"] = pd.to_numeric(ife["ADJ_PRC"], errors="coerce")
    price = wide(ife, "ADJ_PRC").abs()
    d = {"price": price.where(price > 0)}
    numeric = {"ret": "total_return", "market_cap": "size", "sic": "sic"}
    cols = ["permno", "month", *numeric]
    ciz = pd.read_csv(root / "processed/security_monthly_clean.csv.gz", usecols=cols)
    ciz["month"] = pd.PeriodIndex(ciz.month, freq="M")
    if ciz.duplicated(["permno", "month"]).any():
        raise ValueError("Duplicate clean CIZ keys; explicit deduplication required")
    for src, dest in numeric.items():
        ciz[src] = pd.to_numeric(ciz[src], errors="coerce")
        d[dest] = wide(ciz, src, "permno")
    d["size"] = d["size"].where(d["size"] > 0)
    # Restore observed returns omitted from the filtered metadata extract.
    # Keep its capitalization and industry missingness, as documented in the report.
    raw_path = root / "raw/security_monthly.csv.gz"
    end_chunks, return_chunks = [], []
    raw_columns = ["PERMNO", "SecurityEndDt", "YYYYMM", "MthRet", "MthCap"]
    for part in pd.read_csv(raw_path, usecols=raw_columns, chunksize=250_000):
        end_chunks.append(part[["PERMNO", "SecurityEndDt"]].drop_duplicates())
        selected = part.loc[part.PERMNO.isin(ids), ["PERMNO", "YYYYMM", "MthRet", "MthCap"]].copy()
        selected["month"] = pd.PeriodIndex(selected.YYYYMM.astype(str), freq="M")
        selected["MthRet"] = pd.to_numeric(selected.MthRet, errors="coerce")
        selected["MthCap"] = pd.to_numeric(selected.MthCap, errors="coerce")
        return_chunks.append(
            selected.loc[
                selected.month.isin(months), ["PERMNO", "month", "MthRet", "MthCap"]
            ].drop_duplicates()
        )
    raw_monthly = pd.concat(return_chunks, ignore_index=True).drop_duplicates()
    if raw_monthly.duplicated(["PERMNO", "month"]).any():
        raise ValueError("Conflicting raw CIZ returns for a security-month")
    raw_total = wide(raw_monthly, "MthRet")
    clean_total = d["total_return"]
    if ((raw_total - clean_total).abs() > 1e-10).any().any():
        raise ValueError("Raw and clean CIZ monthly returns disagree")
    restored = clean_total.isna() & raw_total.notna()
    d["total_return"] = raw_total.combine_first(clean_total)
    # A final security date is used only in its event month, never as an ex-ante survivor filter.
    ends = pd.concat(end_chunks).drop_duplicates()
    if ends.PERMNO.duplicated().any():
        raise ValueError("Conflicting security end dates")
    end_month = (
        pd.to_datetime(ends.set_index("PERMNO").SecurityEndDt).dt.to_period("M").reindex(ids)
    )
    d["delisted"] = pd.DataFrame(
        months.values[:, None] == end_month.values[None, :], index=months, columns=ids
    )
    d["price_return"] = d["price"].div(d["price"].shift(1)) - 1
    if d["total_return"].lt(-1).any().any():
        raise ValueError("Observed total return below -100%")
    f = read_factors(root / "raw/factor_returns_monthly.csv").reindex(months)
    b = pd.read_csv(root / "raw/market_benchmarks.csv")
    b.index = pd.PeriodIndex(b.month, freq="M")
    market = b.vw_market_return.reindex(months).rename("CRSP_market")
    required = pd.period_range(pd.Period(config.start, "M") - config.window, config.end, freq="M")
    if (
        not required.isin(months).all()
        or f.loc[required].isna().any().any()
        or market.loc[required].isna().any()
    ):
        raise ValueError("Missing input months or factors for requested estimation/holding period")
    audit = {
        "ife_rows": len(ife),
        "ife_securities": len(ids),
        "ciz_rows": len(ciz),
        "ciz_security_overlap_fraction": float(pd.Series(ids).isin(ciz.permno).mean()),
        "total_return_source": "raw CIZ MthRet, including terminal months; clean CIZ only if raw is missing",
        "raw_total_return_path": str(raw_path.relative_to(config.data_dir)),
        "raw_total_return_rows_in_ife_panel": len(raw_monthly),
        "total_return_rows_restored_from_raw": int(restored.to_numpy().sum()),
        "terminal_return_rows_restored_from_raw": int((restored & d["delisted"]).to_numpy().sum()),
        "raw_clean_return_overlap_verified": True,
        "monthly_factors_source": "Fama-French five-factor monthly data, supplied 202607 vintage",
        "capitalization_source": "Filtered monthly metadata; missing fields retained as in the final report",
        "event_date_source": "raw CRSP SecurityEndDt, matched only to realized event month",
        "first_month": str(months[0]),
        "last_month": str(months[-1]),
    }
    return Panel(d, f, market, audit)
