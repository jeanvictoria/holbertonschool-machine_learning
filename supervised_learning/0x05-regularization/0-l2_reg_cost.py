#!/usr/bin/env python3
"""
function that calculates cost of NN using L2
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dictionary of weights and biases (numpy.ndarrays) of NN
    L: num layers in NN
    m: num data points used
    """
    weights_sum = 0
    for key, num in weights.items():   # weights id a key value dict
        if (key[0] == "W"):
            num = weights[key]
            weights_sum += np.linalg.norm(num)
    L2_Cost = (cost + (lambtha / (2 * m)) * weights_sum)
    return(L2_Cost)
