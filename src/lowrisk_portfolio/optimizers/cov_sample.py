"""Sample covariance with T-1 degrees of freedom.

Return matrices have shape (observations, assets); weights are decimal fractions.
"""

import numpy as np


def estimate(X):
    "Return the N-by-N sample covariance (ddof=1) of T-by-N monthly returns."
    return np.cov(X, rowvar=False)
