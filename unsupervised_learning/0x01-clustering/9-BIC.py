#!/usr/bin/env python3
"""
finds the best number of cluster for GMM using Baysian
Information Criterion
"""


import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds best num of clusters for a GMM
    :param X: np.ndarray, (n,d) data set
    :param kmin:pos int, min num clusters to check for inclusive
    :param kmax:pos int, max num clusters to check for inclusive
    :param iterations:pos int, contain max num iterations for EM
    :param tol:non-neg float, caontain tolerance for EM algorithm
    :param verbose:boolean, determines if EM should/should not print
    :return:best_k, best_result, l, b
    """

    if not isinstance(X, np.ndarray) or X.shape != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol <= 0:
        return None, None, None, None
    if not isinstance(vverbose, bool):
        return None, None, None, None

    (n, d) = X.shape
    res_k, results, tot_like, b_bic = [], [], [], []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_like = expectation_maximization(
            X, k, iterations, tol, verbose)
        results.append((pi, m, S))
        res_k.append(k)
        tot_like.append(log_like)
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        bic = p * np.log(n) - 2 * log_like
        b_bic.append(bic)
    b_bic = np.asarray(b_bic)
    best_b = np.argmin(b_bic)
    tot_like = np.asarray(tot_like)
    return (res_k[best_b], results[best_b], tot_like, b_bic)
