#!/usr/bin/env python3
"""
Function calculates the mean and covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """
    calculates the mean and covariance of data set
    :param X: numpy.ndarray, shape(n, d), contains data set
    :param n: num of data points
    :param d: num of dimensions in each data point
    If X is not a 2D numpy.ndarray,
    raise a TypeError with the message X must be a 2D numpy.ndarray
    f n is less than 2,
    raise a ValueError with the message X must contain multiple data points
    :param mean: numpy.ndarray, shape(1, d), contains mean of data set
    :param cov: numpy.ndarray, shape(d, d), contains cov data set
    :return: mean, cov
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0).reshape(1, X.shape[1])
    X = X - mean
    cov = ((np.dot(X.T, X)) / (X.shape[0] - 1))
    return mean, cov
