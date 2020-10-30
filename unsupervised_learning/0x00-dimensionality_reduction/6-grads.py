#!/usr/bin/env python3
""" Function calculates gradients of Y"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    calvulates gradients of Y
    :param Y: np.ndarray, shape (n, ndim)
    :param P: np.ndarray, shape (n, n)
    :return: dY: np.ndarray, shape (n, n)
                contains gradients of Y
            Q: np.ndarray, shape (n, n)
                contains Q affiliates of Y
    """
    # variables
    n, ndim = Y.shape
    Q, num = Q_affnities(Y)
    affDiff = P - Q
    affDiff_term = np.expand_dims((affDiff * num).T, axis=-1)
    dY = np.zeros([n, ndim])
    for i in range(n):
        Y_diff = Y[i, :] - Y
        dY[i, :] = (np.sum((affDiff_term[i, :] * Y_diff), axis=0))
    return (dy, Q)
