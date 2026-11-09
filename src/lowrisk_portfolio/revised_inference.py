"""Render the original-series market and BAB comparisons in Table III, Panel B."""

from pathlib import Path

import numpy as np
import pandas as pd


def render_market_panel(text: str, output: Path) -> str:
    market = pd.read_csv(output / "market_inference.csv").query("block_months == 6")
    market = market.set_index(["period", "market"], verify_integrity=True)
    bab = pd.read_csv(output / "extensions/ext09_alpha_bab/metrics.csv")
    bab = bab.query("variant == 'source_weight'").set_index("period", verify_integrity=True)

    def number(value, places):
        return f"{value:.{places}f}".replace("-", "$-$")

    def interval(row):
        return (
            f"{number(row.delta_sharpe, 3)} "
            f"[{number(row.ci95_lower, 3)}, {number(row.ci95_upper, 3)}]"
        )

    def regression(alpha, stat):
        return f"{number(100 * alpha, 2)} ({number(stat, 2)})"

    anchor = text.index(r"\label{tab:inference}")
    heading = text.index("Panel B. Market-relative Sharpe differences", anchor)
    start = text.index(r"\midrule", heading) + len(r"\midrule")
    end = text.index(r"\bottomrule", start)
    rows = []
    for period, label in [
        ("full", "Full"),
        ("2003_2011", "2003--2011"),
        ("2012_2020", "2012--2020"),
    ]:
        crsp, ff = [market.loc[(period, name)] for name in ["CRSP_market", "FF_market"]]
        fit = bab.loc[period]
        if not crsp.months == ff.months == fit.months:
            raise ValueError(f"Market and factor regressions use different samples: {period}")
        np.testing.assert_allclose(
            [crsp.ff5_alpha_annual, crsp.ff5_alpha_hac_t],
            [fit.ff5_alpha_annual, fit.ff5_alpha_hac_t],
            atol=1e-10,
        )
        primary = interval(crsp)
        alpha = regression(fit.ff5_bab_alpha_annual, fit.ff5_bab_alpha_hac_t)
        if period != "2003_2011":
            style = r"\textbf" if period == "full" else r"\underline"
            primary, alpha = [style + "{" + item + "}" for item in [primary, alpha]]
        rows.append(
            " & ".join(
                [
                    label,
                    str(int(crsp.months)),
                    number(crsp.strategy_sharpe, 3),
                    primary,
                    interval(ff),
                    regression(fit.ff5_alpha_annual, fit.ff5_alpha_hac_t),
                    alpha,
                ]
            )
            + r"\\"
        )
    return text[:start] + "\n" + "\n".join(rows) + "\n" + text[end:]
