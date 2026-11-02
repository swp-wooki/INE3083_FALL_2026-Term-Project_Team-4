"""Correlation eigenvalue clipping at the Marchenko-Pastur upper edge, preserving marginal variances.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np


def estimate(X):
    T, n = X.shape
    S = np.cov(X, rowvar=False)
    s = np.sqrt(np.diag(S))
    C = S / np.outer(s, s)  # Correlation matrix
    lam, U = np.linalg.eigh(C)
    lam_max = (1 + np.sqrt(n / T)) ** 2  # Marchenko-Pastur upper edge
    noise = lam < lam_max
    if noise.any():
        lam = np.where(noise, lam[noise].mean(), lam)  # Flatten noise eigenvalues
    Cc = U @ np.diag(lam) @ U.T
    d = np.sqrt(np.diag(Cc))
    Cc = Cc / np.outer(d, d)  # Renormalize the diagonal to one
    return Cc * np.outer(s, s)
