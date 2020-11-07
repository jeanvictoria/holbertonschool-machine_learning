#!/usr/bin/env python3
"""
calculates the probability density function of a Gaussian distribution
"""


import numpy as np


def pdf(X, m, S):
    """
    caalculates the probability density function
    :param X: np.ndaray, (n, d) data points
    :param m: np.ndarray, (d,) mean of distribution
    :param S: np.ndarray, (d,d) covariance of distribution
    :return: P or None
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1] or X.shape[1] != S.shape[1]:
        return None

    n, d = X.shape
    Xm_diff = (X - m)
    invS = np.linalg.inv(S)
    detS = np.linalg.det(S)
    dp_matVec = np.einsum('...k,kl,...l->...', Xm_diff, invS, Xm_diff)
    Psub1 = (pow(np.sqrt((2*np.pi)**d * detS), -1))
    Psub2 = np.exp(-dp_matVec / 2)
    P = Psub1 * Psub2
    P = np.where(P > 1e-300, P, 1e-300)
    return P
