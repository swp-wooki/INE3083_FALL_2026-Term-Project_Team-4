"""Regression checks against analytic solutions and exhaustive finite references."""

import itertools
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from lowrisk_portfolio.optimizers.erc import (
    constrained_risk_budgeting,
    erc,
    risk_contrib,
)
from lowrisk_portfolio.optimizers.max_sharpe_bs import bayes_stein, max_sharpe_bs
from lowrisk_portfolio.optimizers.mdp import mdp
from lowrisk_portfolio.optimizers.ratio_solver import maximize_ratio


def vertices(n, cap):
    k = int(np.floor(1 / cap + 1e-12))
    residual = 1 - k * cap
    for selected in itertools.combinations(range(n), k):
        w = np.zeros(n)
        w[list(selected)] = cap
        if abs(residual) < 1e-12:
            yield w
        else:
            for j in range(n):
                if j not in selected:
                    result = w.copy()
                    result[j] = residual
                    yield result


class RepairedOptimizerTests(unittest.TestCase):
    def test_mdp_constant_correlation_analytic_solution(self):
        sd = np.array([0.03, 0.07, 0.09, 0.14, 0.2])
        correlation = np.full((len(sd), len(sd)), 0.65)
        np.fill_diagonal(correlation, 1)
        S = correlation * np.outer(sd, sd)
        expected = (1 / sd) / (1 / sd).sum()
        labels = list("ABCDE")
        w = mdp(cov=pd.DataFrame(S, index=labels, columns=labels))
        np.testing.assert_allclose(w, expected, atol=1e-10, rtol=0)
        self.assertEqual(list(w.index), labels)

    def test_positive_ratio_agrees_with_independent_scalar_search(self):
        S = np.array([[0.01, 0.012], [0.012, 0.09]])
        for a in (np.array([0.03, 0.01]), np.array([-0.01, 0.09])):
            for cap in (0.6, 1.0):
                score = lambda x: (a @ [x, 1 - x]) / np.sqrt(np.array([x, 1 - x]) @ S @ [x, 1 - x])
                reference = minimize_scalar(
                    lambda x: -score(x),
                    bounds=(1 - cap, cap),
                    method="bounded",
                    options={"xatol": 1e-14},
                )
                best = max(score(reference.x), score(1 - cap), score(cap))
                w, info = maximize_ratio(S, a, cap, return_info=True)
                self.assertLess(abs(score(w[0]) - best), 1e-10)
                self.assertLess(info["relative_duality_gap"], 1e-8)

    def test_negative_ratio_matches_exhaustive_vertices(self):
        rng = np.random.default_rng(932)
        for n in (4, 6, 8):
            for cap in (0.25, 0.3, 0.6, 1.0):
                if cap * n < 1:
                    continue
                for _ in range(3):
                    A = rng.normal(size=(n, n))
                    S = A.T @ A + 0.1 * np.eye(n)
                    a = -rng.uniform(0.05, 2, n)
                    score = lambda w: a @ w / np.sqrt(w @ S @ w)
                    expected = max(score(w) for w in vertices(n, cap))
                    w = maximize_ratio(S, a, cap)
                    self.assertAlmostEqual(score(w), expected, places=10)

    def test_nonpositive_feasible_mean_can_include_positive_asset_means(self):
        S = np.diag([1.0, 2, 4, 3, 5, 1.5])
        a = np.array([0.1, -0.5, -0.6, -0.7, -0.8, -0.9])
        cap = 0.3
        expected = max(a @ w / np.sqrt(w @ S @ w) for w in vertices(len(a), cap))
        w = maximize_ratio(S, a, cap)
        self.assertAlmostEqual(a @ w / np.sqrt(w @ S @ w), expected, places=12)

    def test_zero_maximum_and_singleton_feasible_set(self):
        S = np.eye(4)
        a = np.array([1.0, -1, -1, -1])
        w = maximize_ratio(S, a, 0.5)
        self.assertEqual(a @ w, 0)
        np.testing.assert_array_equal(maximize_ratio(S, a, 0.25), np.full(4, 0.25))
        np.testing.assert_array_equal(erc(cov=S, cap=0.25), np.full(4, 0.25))

    def test_box_risk_budgeting_diagonal_analytic_solution(self):
        sd = np.array([0.05, 0.1, 0.2, 0.3, 0.5])
        cap = 0.3
        multiplier = brentq(lambda z: np.minimum(cap, z / sd).sum() - 1, 0, 1)
        expected = np.minimum(cap, multiplier / sd)
        w, info = constrained_risk_budgeting(np.diag(sd * sd), cap, return_info=True)
        np.testing.assert_allclose(w, expected, atol=1e-9, rtol=0)
        self.assertLess(info["budget_residual"], 1e-10)

    def test_box_risk_budgeting_preserves_free_risk_budgets(self):
        # Richard--Roncalli (2019), Example 1, PDF p.6. Compare the defining
        # KKT property, not their Table 2's deliberately rejected least squares.
        sd = np.array([0.1, 0.15, 0.2, 0.3])
        correlation = np.full((4, 4), 0.5)
        np.fill_diagonal(correlation, 1)
        correlation[2, 3] = correlation[3, 2] = 0.75
        S = correlation * np.outer(sd, sd)
        original = erc(cov=S)
        np.testing.assert_allclose(original, [0.4101, 0.2734, 0.1899, 0.1266], atol=5e-5, rtol=0)
        w, info = constrained_risk_budgeting(S, 0.3, return_info=True)
        rc = w * (S @ w)
        free = w < 0.3 - 1e-9
        np.testing.assert_allclose(rc[free], info["alpha"] / 4, atol=1e-10, rtol=0)
        self.assertTrue(np.all(rc[~free] <= info["alpha"] / 4 + 1e-10))
        self.assertLess(np.ptp(risk_contrib(w, S)[free]), 1e-9)

    def test_bayes_stein_original_equations_and_layout_invariance(self):
        rng = np.random.default_rng(771)
        X = rng.normal(0.01, 0.03, size=(36, 8))
        # Independent scalar contractions of Jorion eqs. (14), (17), (18).
        mean = np.mean(X, axis=0)
        adjusted = np.cov(X, rowvar=False) * (35 / (36 - 8 - 2))
        one = np.ones(8)
        precision_mean = np.linalg.solve(adjusted, mean)
        precision_one = np.linalg.solve(adjusted, one)
        target = np.sum(precision_mean) / np.sum(precision_one)
        d = mean - target
        phi = 10 / (10 + 36 * np.dot(d, np.linalg.solve(adjusted, d)))
        np.testing.assert_allclose(
            bayes_stein(X), (1 - phi) * mean + phi * target, atol=1e-14, rtol=0
        )
        C = 0.0005 * np.eye(8) + 0.5 * np.cov(X, rowvar=False)
        for rf in (0, 0.05):
            wc = max_sharpe_bs(np.array(X, order="C"), cov=C, cap=0.3, rf=rf)
            wf = max_sharpe_bs(np.array(X, order="F"), cov=C, cap=0.3, rf=rf)
            np.testing.assert_array_equal(wc, wf)

    def test_invalid_inputs_fail_instead_of_returning_weights(self):
        with self.assertRaises(ValueError):
            maximize_ratio(np.eye(4), np.ones(4), cap=0.2)
        with self.assertRaises(ValueError):
            maximize_ratio(np.diag([1.0, -1]), np.ones(2))
        with self.assertRaises(ValueError):
            constrained_risk_budgeting(np.array([[1.0, np.nan], [np.nan, 1.0]]))


if __name__ == "__main__":
    unittest.main()
