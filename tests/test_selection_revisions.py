"""Checks on inference that must survive changes to the underlying input panel."""

import unittest

import numpy as np
import pandas as pd

from lowrisk_portfolio.selection_revisions import _blocks, _interaction, _sr, bootstrap_precision


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


if __name__ == "__main__":
    unittest.main()
