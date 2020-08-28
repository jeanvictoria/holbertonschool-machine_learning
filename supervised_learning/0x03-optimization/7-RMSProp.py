#!/usr/bin/env python3
"""
function updates var using RMSProp optimization
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    alpha: learning rate
    beta2: the RMSProp weight
    epsilon: small number to avoid division by zero
    var: numpy.ndarray containing variable to update
    grad: numpy.ndarray containing gradient of var
    s: previous second moment of var
    """

    Sdvar = beta2 * s + (1 - beta2) * (grad**2)
    W_upvar = var - alpha*(grad / (Sdvar**(1/2) + epsilon))
    return W_upvar, Sdvar
