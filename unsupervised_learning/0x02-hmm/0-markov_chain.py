#!/usr/bin/env python3
"""
Function determines the probability of a MArkov Chain being
in a particular state after a spcified num of operations
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    determines if markov chain isin a particilar state
    after iterations
    :param P: square 2D np.ndarray, (n, n) transition matrix
    P[i, j]:the probability of transitioning from state i to j
    n : num of states in markov chain
    s: np.ndarray, (1, n), prob of starting in each state
    t: num of iterations markov chain has been through
    :return:  np.ndarray, (1, n) prob of being in specific state
                agfter t interations, or None
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] < 1 or P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if s.shape[1] != P.shape[0] or s.shape[0] != 1:
        return None
    if type(t) != int or t < 0:
        return None
    P = np.linalg.matrix_power(P, t)
    state_prob = np.matmul(s, P)
    return state_prob
