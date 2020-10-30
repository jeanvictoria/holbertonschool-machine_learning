#!/usr/bin/env python3
"""
function performs PCA on a dataset
"""

import numpy as np


def pca(X, var=0.95):

    """
    function performs PCA on a dataset
    :param X: numpy.ndarray, shape(n, d)
    :param n: num data points
    :param d: num dimensions in each point
    :param var: fraction of variance the PCA transformation should maintain
    :param W: numpy.ndarray, shape(d, nd) nd is new dimensionality of
                transformed X
    :return: weights matrix; maintains var fraction of X orig variance
    """

    # get s and Vt values
    U, s, Vt = np.linalg.svd(X)

    # calculate cumulative variance
    s_sum = np.sum(s)
    tot_sum = np.cumsum(s)

    # threshold, variance included in s
    cum_var = tot_sum / s_sum

    # find indices that are non-zero, grouped by element
    r = np.argwhere(cum_var >= var)[0, 0]

    W = Vt[:r + 1].T
    return (W)
