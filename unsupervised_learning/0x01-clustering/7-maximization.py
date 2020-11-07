#!/usr/bin/env python3
"""
Calculates the maximization step in the EM algoritm for GMM
"""
import numpy as np


def maximization(X, g):
    """
    updating the clusters
    :param X: np.ndarray, (n, d), data set
    :param g: np.ndarray, (k, n), posterior probs
                for eah data point in each cluster
    :return: pi, m, S, or None, None, None
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape

    prob_sum = np.sum(g, axis=0)
    prob_tot = np.ones((n, ))
    # remember to use np.isclose
    if not np.isclose(prob_sum, prob_tot).all():
        return None, None, None

    k = g.shape[0]
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for idx in range(k):
        g_sum = np.sum(g[idx], axis=0)

        # calculate & update m
        m[idx] = np.sum((g[idx, :, np.newaxis] * X), axis=0) / (g_sum)

        # update  covariance
        cov_sum = np.dot(g[idx] * (X - m[idx]).T, (X - m[idx]))
        S[idx] = cov_sum / np.sum(g[idx])

        # calculate & update pi
        pi[idx] = np.sum(g[idx]) / n

    return (pi, m, S)
