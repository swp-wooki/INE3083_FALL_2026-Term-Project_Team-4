"""Ridge covariance: add lambda times mean sample variance to the diagonal.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np

LAMBDA = 0.3


def estimate(X, lam=LAMBDA):
    S = np.cov(X, rowvar=False)
    n = len(S)
    return S + lam * (np.trace(S) / n) * np.eye(n)
