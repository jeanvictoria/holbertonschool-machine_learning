#!/usr/bin/env python3
"""
Function performs K-means on a dataset
"""


import numpy as np


def initialize(X, k):
    """
    Initialize K clusters, mult variable dist
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0 or k > X.shape[0]:
        return None
    n, d = X.shape
    min_X = X.min(axis=0)
    max_X = X.max(axis=0)
    return np.random.uniform(min_X, max_X, size=(k, d))


def kmeans(X, k, iterations=1000):
    """
    performs K-means on a dataset
    :param X: np.ndarray, shape(n, d)
              containing dataset
    :param k: positive integer containing num of clusters
    :param iterations: 1000
    :return: C, clss or None, None on failure
    """

    centroids = initialize(X, k)

    if type(iterations) != int or iterations <= 0:
        return None, None
    if centroids is None:
        return None, None
    # clustering
    for m in range(iterations):
        centCopy = np.copy(centroids)
        dists = np.sqrt((X - centroids[:, np.newaxis])**2).sum(axis=-1)
        klass = np.argmin(dists, axis=0)

        for k in range(centroids.shape[0]):
            if (X[klass == k].size == 0):
                centroids[k, :] = np.random.uniform(X.min(axis=0),
                                                    X.max(axis=0), size=(1, d))
            else:
                centroids[k, :] = (X[klass == k].mean(axis=0))
        if (centCopy == centroids).all():
            return (centroids, klass)

    return (centroids, klass)
