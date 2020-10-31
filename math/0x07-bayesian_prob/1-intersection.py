#!/usr/bin/env python3
"""
calculates the intersection of obtaining this data
with the various hypothetical probabilities
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    calculates the intersection of obtaining this data
    with the various hypothetical probabilities
    :param x: num of patients that develope side effects
    :param n: tot num of patients observed
    :param P: 1D np.ndarray containing prior beliefs
    :param Pr: 1D np.ndarray containing prior beliefs
    :return: 1D np.ndarray containing intersection of
    obtaining x and n  with each probability in P.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        m = "x must be an integer that is greater than or equal to 0"
        raise ValueError(m)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if np.amin(P) < 0 or np.amax(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.amin(Pr) < 0 or np.amax(Pr) > 1:
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    factor = np.math.factorial
    factNX = (factor(n) / (factor(x) * factor(n - x)))
    likelihood = factNX * (P**x) * ((1 - P)**(n - x))
    return likelihood * Pr
