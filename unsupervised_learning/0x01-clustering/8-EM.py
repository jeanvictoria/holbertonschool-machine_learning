#!/usr/bin/env python3
"""performs expectation maximization for a GMM"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """
    perfors the expectation maxximizatin for GMM
    :param X: np.ndarray, (n, d), data set
    :param k: pos int, num clusters
    :param iterations: pos int, max num interations for
                        algorithm
    :param tol: non-neg float, tolerancee of log likelihood
    :param verbose: boolean, determines print or not algorithm
                    info
    :return: pi, m, S, g, l, or None, None, None, None, None
    """

    if not isinstance(X, np.ndarray) or X.shape != 2:
        return None, None, None, None, None, None
    if type(k) != int or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None, None

    # initialize pi, mean, covariance
    pi, m, S = intialize(X, k)

    prevL = 0
    for i in range(iterations):

        # calculate probability at each data point, E-step
        g, logL = expectation(X, pi, m, S)

        # calculate maimizatin step, each step
        pi, m, S = maximization(X, g)

        if (verbose is True):
            if (i % 10 == 0) or (i == 0):
                print("Log Likelihood after {} iterations: {}".format(i, logL))
            if (i == interations - 1):
                print("Log Likelihood after {} iterations: {}".format(i, logL))
            if abs(logL - prevL) <= tol:
                print("Log Likelihood after {} iterations: {}".format(i, logL))
                break
        if abs(logL - prevL) <= tol:
            break
        prevL = logL
    return (pi, m, S, g, logL)
