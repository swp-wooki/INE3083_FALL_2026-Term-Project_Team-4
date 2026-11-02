"""Elton-Gruber constant-correlation covariance with sample marginal variances.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np


def estimate(X):
    S = np.cov(X, rowvar=False)  # Sample covariance (ddof=1)
    n = len(S)
    s = np.sqrt(np.diag(S))  # Marginal standard deviations
    C = S / np.outer(s, s)  # Sample correlation matrix
    rbar = (C.sum() - n) / (n * (n - 1))  # Mean off-diagonal correlation; subtract n diagonal ones
    Ct = np.full((n, n), rbar)
    np.fill_diagonal(Ct, 1.0)  # Constant-correlation matrix
    return Ct * np.outer(s, s)  # Rescale to covariance
