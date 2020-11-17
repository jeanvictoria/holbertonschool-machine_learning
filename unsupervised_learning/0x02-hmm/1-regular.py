#!/usr/bin/env python3
"""
Function determines the steady state prob
of a regular markov chain
"""

import numpy as np


def regular(P):
    """
    Determines steady state of regular markov chain
    :param P: np.ndarray, shape (n, n), transition matrix
    n: num states in Markov chain
    :return: np.ndarray, shape(1, n) contain state prob
            or None
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1] or P.shape[0] < 1:
        return None
    if (np.where(P < 0, 1, 0).any()):
        return None
    # if np.sum(P, axis=1).all() != 1:
        # return None
    n = P.shape[0]
    states = [P]
    probNow = P

    # where this does not exist
    while np.where(probNow != 0, 0, 1).any():
        probNow = np.dot(P, probNow)
        # to solve the bool error add for statement
        # on the end
        if any(np.allclose(probNow, i) for i in states):
            return None
            # fill probStates array wil the current probability
        states.append(probNow)
    while True:
        probPrev = probNow
        # could also use matmul here
        probNow = np.dot(P, probNow)
        if np.array_equal(probNow, probPrev):
            return probNow[0:1]
