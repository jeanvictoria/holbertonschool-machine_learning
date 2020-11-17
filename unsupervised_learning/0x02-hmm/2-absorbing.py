#!/usr/bin/env python3
"""
Function determines if a markov chain is absorbing
"""

import numpy as np


def absorbing(P):
    """
    determines if a markov chain is an absorbing chain
    :param P: np.ndarray, (n,n), transition matrix
    n: num of states in the markov chain
    :return: True  or False
    """
    if not isinstance(P, np.ndarray) or (len(P.shape) != 2):
        return False
    if P.shape[0] != P.shape[1] or P.shape[0] < 1:
        return False
    if ((np.where(P < 0, 1, 0).any()
         or not np.where(np.isclose(P.sum(axis=1), 1), 1, 0).any())):
        # long if statement use double parens
        return False

    if (np.all(np.diag(P) == 1)):
        # absorbing proven if matrix diagnol all = 1
        return True

    if not (np.any(np.diagonal(P) == 1)):
        # if the diagnol returned is not all 1s return False
        return False

    for i in (range(P.shape[0])):3
        # account for the transitioning from state i to state j
        for j in (range(P.shape[1])):
            # verify above NUll does not apply
            if ((i == j) and (i + 1 < len(P))):
                # prob of transitionin from i to j and j to i
                if (P[i][j + 1] == 0 and P[i + j][j] == 0):
                    return False
    return True
