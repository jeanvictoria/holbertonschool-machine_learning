#!/usr/bin/env python3
"""
function calculats likelihood of obtaining data
given various hypothetical probabilities of developing
severe side effects
"""

import numpy as np


def likelihood(x, n, P):
    """calculats likelihood
    :param x: num patients that develop
    :param n: total number of patients observed
    :param p: np.ndarray, contains hypothetical probabilities
    Return: numpy.ndarray: containing likelihood of obtaining data
            x and n, for each probability in P
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        m = "x must be an integer that is greater than or equal to 0"
        raise ValueError(m)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.min(P) < 0 or np.amax(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
    factor = np.math.factorial
    factNX = (factor(n) / (factor(x) * factor(n - x)))
    return factNX * (P**x) * ((1 - P)**(n - x))
