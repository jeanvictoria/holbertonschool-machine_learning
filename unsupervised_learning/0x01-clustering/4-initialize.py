#!/usr/bin/env python3
"""
initialize variables for a Gaussian Mixture Model
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    initialize variuables
    :param X: np.ndarray, (n, d) data set
    :param k: pos int, num clusters
    :return: pi, m, S, or None, None, None
    """

    centroids = kmeans(X, k)[0]
    if centroids is None:
        return None, None, None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] <= k:
        return None, None, None
    n, d = X.shape
    m = centroids
    pi = np.repeat((1/k,), k)
    S = np.tile(np.identity(d), (k, 1))
    S = np.reshape(S, (k, d, d))
    return (pi, m, S)
