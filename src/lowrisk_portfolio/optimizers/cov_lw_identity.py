"""Ledoit-Wolf linear shrinkage toward a scaled identity matrix.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np


def estimate(X):
    T, n = X.shape
    Y = X - X.mean(axis=0)
    S = Y.T @ Y / T
    mu = np.trace(S) / n
    F = mu * np.eye(n)  # Target: mean variance times identity
    d2 = ((S - F) ** 2).sum()
    # b̄² = 1/T² Σ_t ‖y_t y_t' − S‖² = 1/T² [Σ_t (y_t'y_t)² ] − ‖S‖²/T
    b2_bar = ((Y**2).sum(axis=1) ** 2).sum() / T**2 - (S**2).sum() / T
    b2 = min(b2_bar, d2)
    shrink = b2 / d2 if d2 > 0 else 1.0
    return shrink * F + (1 - shrink) * S
