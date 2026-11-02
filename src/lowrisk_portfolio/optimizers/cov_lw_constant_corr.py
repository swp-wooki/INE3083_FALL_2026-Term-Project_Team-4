"""Ledoit-Wolf linear shrinkage toward constant correlation. The final experiment uses legacy=False.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np

LEGACY = False


def estimate(X, legacy=None):
    legacy = LEGACY if legacy is None else legacy
    T, n = X.shape
    Y = X - X.mean(axis=0)  # Demean observations
    S = Y.T @ Y / T  # Sample covariance with denominator T
    var = np.diag(S)
    sd = np.sqrt(var)

    # Constant-correlation target F
    rbar = ((S / np.outer(sd, sd)).sum() - n) / (n * (n - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)

    # pi: sum of entrywise asymptotic variances
    Y2 = Y**2
    pi_mat = Y2.T @ Y2 / T - S**2
    pi_hat = pi_mat.sum()

    # rho: diagonal pi terms plus weighted off-diagonal theta terms
    theta = (Y**3).T @ Y / T - var[:, None] * S  # theta_ii,ij (row i, column j)
    np.fill_diagonal(theta, 0.0)
    if legacy:  # Legacy estimator convention
        ratio = np.sqrt(np.outer(1 / sd, sd))
        rho_hat = np.trace(pi_mat) + (rbar / 2) * (ratio * theta + ratio.T * theta.T).sum()
    else:  # Literature convention
        ratio = np.outer(1 / sd, sd)  # [i,j] = σ_j / σ_i
        rho_hat = np.trace(pi_mat) + rbar * (ratio * theta).sum()

    # Squared distance between target and sample
    gamma_hat = ((F - S) ** 2).sum()

    delta = max(0.0, min((pi_hat - rho_hat) / gamma_hat / T, 1.0))
    return delta * F + (1 - delta) * S
