"""Quadratic inverse shrinkage following Ledoit and Wolf (2022). Handles singular samples and preserves the sample trace.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np


def estimate(X):
    T, p = X.shape
    Y = X - X.mean(axis=0)
    n = T - 1  # One degree of freedom is lost by demeaning
    c = p / n  # Concentration ratio
    S = Y.T @ Y / n
    S = (S + S.T) / 2
    lam, U = np.linalg.eigh(S)  # Ascending eigenvalues
    lam = np.clip(lam, 0, None)

    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35  # Kernel bandwidth
    k = min(p, n)  # Number of nonzero eigenvalues
    inv = 1.0 / np.maximum(lam[p - k :], 1e-12)  # Inverse eigenvalues
    Li = np.tile(inv, (k, 1)).T  # Li[i,j] = inv_i
    Dij = Li - Li.T  # inv_i − inv_j
    den = Dij**2 + (Li * h) ** 2
    theta = (Li * Dij / den).mean(axis=0)  # Real part of smoothed Stieltjes transform
    Htheta = (Li * Li * h / den).mean(axis=0)  # Hilbert transform
    A2 = theta**2 + Htheta**2

    if p <= n:
        delta = 1 / ((1 - c) ** 2 * inv + 2 * c * (1 - c) * inv * theta + c**2 * inv * A2)
    else:
        d0 = 1 / ((c - 1) * inv.mean())  # Common replacement for zero eigenvalues
        delta = np.concatenate([np.repeat(d0, p - n), 1 / (inv * A2)])

    delta = delta * (lam.sum() / delta.sum())  # Preserve the sample trace
    return U @ np.diag(delta) @ U.T
