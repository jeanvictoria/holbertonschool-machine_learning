#!/usr/bin/env python3
"""
Initialize cluster centroids for K-means
"""

import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroid for K-means
    :param X: np.ndarray, shape(n, d)
                contains data set used for K-means
    :param k: pos integer containing the num clusters
    :return: np.ndarray, of shape (k, d)
    """

    if type(X) != np.ndarray or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None
    n, d = X.shape

    min_X = X.min(axis=0)
    max_X = X.max(axis=0)
    return np.random.uniform(min_X, max_X, size=(k, d))
