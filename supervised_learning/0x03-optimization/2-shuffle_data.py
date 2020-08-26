#!/usr/bin/env python3
"""function shuffles data points in 2 matrices the same way"""


import numpy as np


def shuffle_data(X, Y):
    """
    X first numpy.ndarray shape (m, nx) to shuffle
    Y second numpy.ndarray shape(m, ny) to shuffle
    m num data points
    nx num features in X
    ny num features in Y
    """

    shuffled = np.random.permutation(X.shape[0])
    return X[shuffled], Y[shuffled]
