#!/usr/bin/env python3
"""
Function calculates Q affinities
"""
import numpy as np


def Q_affinities(Y):
    """
    function calculates Q afinities
    :param Y: np.ndarray, shape(n, ndim)
    :return: Q: np.noarray, shape (n,n) contains Q affinities
            num: np.ndarry, shape(n,n) contains numerator of Q affinities
    """

    n, ndim = Y.shape
    sum_Y = np.sum(np.square(Y), 1)
    dist = (np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
    num = (1 + dist)**(-1)
    np.fill_diagonal(num, 0.)
    Q = num / np.sum(num)
    return Q, num
