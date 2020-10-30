#!/usr/bin/env python3
"""
function that peforms  PCA  on a dataset
"""

import numpy as np


def pca(X, ndim):
    """
    function performs PCA on a dataset
    :param X: numpy.ndarray, shape(n, d)
    :param n: num data points
    :param d: num dimensions in each point
    :param ndim: new dimensionality of the transformed X
    :param T: numpy.ndarray, shape(n, ndim) ndim is new dimensionality of
                transformed X
    :return: T
    """
    Xm = X - np.mean(X, axis=0)
    u, s, Vt = np.linalg.svd(Xm)
    W = Vt[:ndim].T
    M = np.matmul(Xm, (W))
    return M
