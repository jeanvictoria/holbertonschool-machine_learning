#!/usr/bin/env python3
"""
Function calculates a correlation matrix
"""
import numpy as np


def correlation(C):
    """
    calculates a correlation matrix
    :param C: np.ndarray, shape(d, d), containing cov matrix
    :param d: num of dimensions
    If C is not a numpy.ndarray,
    raise a TypeError with the message C must be a numpy.ndarray
    If C does not have shape (d, d),
    raise a ValueError with the message C must be a 2D square matrix
    :return: correlation matrix np.ndarray, shape(d, d)
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if (len(C.shape) != 2):
        raise ValueError("C must be a 2D square matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # using the diagonal of covariance matrix
    corr_x = (np.sqrt(np.diag(C)))
    corr_y = corr_x
    corr_matrix = C / (np.outer(corr_x, corr_y))
    return corr_matrix
