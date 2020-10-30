#!/usr/bin/env python3
"""
calculates the cost of the t-SNE transformation
"""

import numpy as np


def cost(P, Q):
    """
    calculates cost of t-SNE transformation
    :param P: np.ndarray, shape(n,n) contains P affinities
    :param Q: np.ndarray, shape(n,n) contains Q affinities
    :return: C: cost of the transformation
    """
    Qmin = Q[Q > 0].min()
    Qval = np.where(Q != 0, Q, Qmin)
    Pmin = P[P > 0].min()
    Pval = np.where(P != 0, P, Pmin)
    quot_PQ = P / Q
    C = (np.sum(np.sum(P * np.log(quot_PQ))))
    return C
