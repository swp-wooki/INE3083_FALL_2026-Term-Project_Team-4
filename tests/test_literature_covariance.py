import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lowrisk_portfolio.optimizers import factor_cov, min_cvar, norm_gmvp


class LiteratureCovarianceTests(unittest.TestCase):
    def test_redundant_norm_constraint_is_exactly_base_gmvp(self):
        # With cap=2/n, sum(w**2)<=2/n follows from budget and box constraints.
        rng = np.random.default_rng(104)
        x = rng.normal(size=(36, 30)) * np.linspace(0.01, 0.15, 30)
        covariance = np.cov(x, rowvar=False)
        base = factor_cov.gmvp(covariance, cap=2 / 30)
        constrained = norm_gmvp.norm_gmvp(cov=covariance, min_eff=15, cap=2 / 30)
        np.testing.assert_array_equal(constrained, base)

    def test_minimum_feasible_norm_returns_exact_equal_weights(self):
        covariance = np.diag([0.01, 0.04, 0.09, 0.16])
        result = norm_gmvp.norm_gmvp(cov=covariance, min_eff=4, cap=0.5)
        np.testing.assert_array_equal(result, np.full(4, 0.25))

    def test_cvar_fractional_tail_matches_lp_objective(self):
        # Six scenarios and a25% tail include the worst loss plus half the next.
        x = np.array(
            [
                [-0.2, -0.03],
                [-0.1, -0.02],
                [0.1, 0.03],
                [0.2, 0.01],
                [0.05, -0.01],
                [0.01, 0.02],
            ]
        )
        captured = []
        original = min_cvar.linprog

        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result.fun)
            return result

        with patch.object(min_cvar, "linprog", side_effect=record):
            weights = min_cvar.min_cvar(x, cap=0.8, alpha=0.25)
        self.assertAlmostEqual(min_cvar.cvar(weights, x, alpha=0.25), captured[0], places=12)
        loss = np.sort(-x @ weights)[::-1]
        self.assertAlmostEqual(captured[0], (loss[0] + 0.5 * loss[1]) / 1.5, places=12)

    def test_failed_cvar_solver_cannot_silently_change_strategy(self):
        failure = SimpleNamespace(success=False, message="deliberate test failure")
        with patch.object(min_cvar, "linprog", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "deliberate test failure"):
                min_cvar.min_cvar(np.array([[0.1, 0.2], [-0.1, 0.3]]))

    def test_invalid_tail_probability_is_rejected(self):
        x = np.array([[0.1, 0.2], [-0.1, 0.3]])
        for alpha in [0, -0.1, 1.1, float("nan")]:
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    min_cvar.min_cvar(x, alpha=alpha)
                with self.assertRaises(ValueError):
                    min_cvar.cvar([0.5, 0.5], x, alpha=alpha)


if __name__ == "__main__":
    unittest.main()
