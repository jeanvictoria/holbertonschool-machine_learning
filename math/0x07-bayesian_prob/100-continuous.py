#!/usr/bin/env python3
"""
 calculates the marginal probability of obtaining the data
"""
import numpy as np
from scipy import math, special


def intersection(x, n, P, Pr):
    """
    calculates the intersection of obtaining this data
    with the various hypothetical probabilities
    """

    factor = math.factorial
    factNX = (factor(n) / (factor(x) * factor(n - x)))
    likelihood = factNX * (P**x) * ((1 - P)**(n - x))
    return likelihood * Pr


def marginal(x, n, P, Pr):
    """
     calculates the marginal probability of obtaining the data
    """
    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, P, Pr):
    """
    calculates the posterior probability for the various
    hypothetical probabilities of developing severe side
    effects given the data
    """
    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)


def posterior(x, n, p1, p2):
    """
    calculates the posterior probability that the probability
    of developing severe side effects falls within a specific
    range given the data:
    :param x: num of patients with sever side effects
    :param n: tot num patients observed
    :param p1: lower bound range
    :param p2: upper bound range
    :return: posterior probability that p is within the range
    [p1, p2] given x and n
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        m = "x must be an integer that is greater than or equal to 0"
        raise ValueError(m)
    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    inter = intersection(x, n, p1, p2)
    return ((special.expn(inter, p2)) / (special.expn(inter, p1)))
