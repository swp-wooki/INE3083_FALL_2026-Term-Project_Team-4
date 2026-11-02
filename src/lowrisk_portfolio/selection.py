"""Validate fixed stock selections and adjust multiple comparisons."""

import numpy as np
import pandas as pd


def load_selection(path, expected_n=None):
    """Validate keys and timing, preserving supplied constituents and ranking exactly."""
    s = pd.read_csv(path)
    required = {"formation_month", "holding_month", "PERMNO", "selection_rank"}
    if not required.issubset(s.columns):
        raise ValueError(f"Missing selection columns: {required - set(s.columns)}")
    s["formation_month"] = pd.PeriodIndex(s.formation_month, freq="M")
    s["holding_month"] = pd.PeriodIndex(s.holding_month, freq="M")
    if s[list(required)].isna().any().any():
        raise ValueError("Missing selection keys")
    if not s.holding_month.eq(s.formation_month + 1).all():
        raise ValueError("Every holding month must equal formation month + 1")
    if s.duplicated(["holding_month", "PERMNO"]).any():
        raise ValueError("Duplicate security-month selection")
    groups = s.groupby("holding_month")
    if s.empty:
        raise ValueError("Selection must not be empty")
    if expected_n is None:
        expected_n = int(groups.size().iloc[0])
    if expected_n < 2:
        raise ValueError("Selection requires at least two stocks per month")
    if not groups.size().eq(expected_n).all():
        raise ValueError(
            f"Each month must contain exactly {expected_n} stocks; no dropping/substitution permitted"
        )
    if not groups.selection_rank.apply(lambda x: sorted(x) == list(range(1, expected_n + 1))).all():
        raise ValueError("Ranks must be exactly 1..N in each month")
    months = pd.PeriodIndex(sorted(s.holding_month.unique()), freq="M")
    if not months.equals(pd.period_range(months[0], months[-1], freq="M")):
        raise ValueError("Missing selection month")
    return s.sort_values(["holding_month", "selection_rank", "PERMNO"]).reset_index(drop=True)


def holm_adjust(pvalues):
    """Holm step-down adjustment across the non-EW models within one period/scenario."""
    p = np.asarray(pvalues, float)
    result = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    if len(order):
        result[order] = np.minimum(
            1, np.maximum.accumulate(p[order] * (len(order) - np.arange(len(order))))
        )
    return result
