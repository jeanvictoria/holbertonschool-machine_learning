#!/usr/bin/env python3
"""
function that initializes all varaibles required to calculate
P affinities in t-SNE
"""
import numpy as np


def P_init(X, perplexity):
    """
    initializes all variables required to calculate the P affinities in t-SNE
    :param X: np.ndarray, shape(n,d)
    :param perplexity:
    :return:
    """
    (n, d) = X.shape
    sumX = np.sum(np.square(X), axis=1)
    D = (np.add(np.add(-2 * np.dot(X, X.T), sumX).T, sumX))
    P = np.zeros([n, n], dtype='float64')
    betas = np.ones([n, 1], dtype='float64')
    H = np.log2(perplexity)

    return (D, P, betas, H)
