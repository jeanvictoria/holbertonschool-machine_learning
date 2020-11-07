#!/usr/bin/env python3
""" Calculate the total intra-cluster variance for a data set"""


import numpy as np


def variance(X, C):
    """ calculate totoal intra-cluster variance for a data set"""

    n, d = X.shape
    k = C.shape[0]

    if ((not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray))):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if d != C.shape[1]:
        return None
    if k >= n:
        return None

    try:
        dist = ((X - C[:, np.newaxis])**2)
        dist = np.cumsum(dist, axis=0)
        dist = np.sqrt(dist)
        var = np.sum((dist)**2)
        return var
    except Exception:
        return None
