"""Checks on inference that must survive changes to the underlying input panel."""

import unittest

import numpy as np
import pandas as pd

from lowrisk_portfolio.selection_revisions import (
    _blocks,
    _interaction,
    _sr,
    bootstrap_precision,
    industry_effects,
    industry_group,
    residual_coskewness,
)


class SelectionInferenceTests(unittest.TestCase):
    def test_chunked_bootstrap_matches_direct_paired_draws(self):
        rng = np.random.default_rng(132)
        frame = pd.DataFrame(rng.normal(size=(37, 4)), columns=["BVC", "BVCP", "BVCL", "RF"])
        ix = _blocks(37, np.random.default_rng(22), draws=73)
        baseline = _sr((frame.BVC - frame.RF).to_numpy()[ix], axis=1)
        rows = bootstrap_precision(frame, draws=73, seed=22, chunk=9).set_index("model")
        for model in ("BVCL", "BVCP"):
            effect = _sr((frame[model] - frame.RF).to_numpy()[ix], axis=1) - baseline
            np.testing.assert_allclose(
                rows.loc[model, ["ci_lo", "ci_hi"]], np.quantile(effect, [0.025, 0.975])
            )

    def test_joint_interaction_matches_separate_period_estimates(self):
        rng = np.random.default_rng(88)
        f = rng.normal(size=(100, 5))
        x = np.column_stack([np.ones(100), f])
        post = np.arange(100) >= 52
        y = x @ rng.normal(size=6) + post * (x @ rng.normal(size=6)) + rng.normal(size=100)
        first = np.linalg.lstsq(x[~post], y[~post], rcond=None)[0]
        last = np.linalg.lstsq(x[post], y[post], rcond=None)[0]
        change = _interaction(y, f, post.astype(float), 4)
        np.testing.assert_allclose(
            [change["alpha_change"] / 1200, change["market_change"], change["cma_change"]],
            (last - first)[[0, 1, 5]],
        )

    def test_residual_coskewness_matches_direct_moments(self):
        rng = np.random.default_rng(5)
        m = pd.Series(rng.normal(0.01, 0.05, 80))
        x = pd.DataFrame({"a": 0.8 * m + rng.normal(0, 0.03, 80), "b": rng.normal(0, 0.06, 80)})
        score = residual_coskewness(x, m)
        for name in x:
            xs, ms = x[name].to_numpy()[-60:], m.to_numpy()[-60:]
            xc, mc = xs - xs.mean(), ms - ms.mean()
            beta = (xc * mc).mean() / (mc * mc).mean()
            e = xc - beta * mc
            direct = -(e * mc * mc).mean() / (np.sqrt((e * e).mean()) * (mc * mc).mean())
            np.testing.assert_allclose(score[name].iloc[-1], direct, rtol=1e-9)

    def test_industry_effects_add_to_return_gap(self):
        rng = np.random.default_rng(9)
        names = list("abcdefgh")
        groups = pd.Series(list("xxyyzzzw"), index=names)
        returns = pd.Series(rng.normal(0, 0.05, 8), index=names)
        bench = pd.Series(rng.uniform(1, 2, 8), index=names)
        bench /= bench.sum()
        port = pd.Series([0.3, 0, 0.2, 0, 0.25, 0.25, 0, 0], index=names)
        effects = industry_effects(port, bench, returns, groups)
        np.testing.assert_allclose(
            (effects.allocation + effects.selection).sum(),
            (port * returns).sum() - (bench * returns).sum(),
        )

    def test_industry_groups(self):
        cases = {
            6798: "Financial: REIT",
            6021: "Financial: banks",
            6331: "Financial: insurance",
            4911: "Utilities",
            2834: "Pharmaceuticals and health",
            7372: "Technology and telecom",
            9999: "Other",
            np.nan: "Other",
        }
        for code, label in cases.items():
            self.assertEqual(industry_group(code), label)


if __name__ == "__main__":
    unittest.main()
