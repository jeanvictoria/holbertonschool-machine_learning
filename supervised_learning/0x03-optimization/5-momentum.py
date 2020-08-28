#!/usr/bin/env python3
"""
function updates variable using grad descent
with momentum optimization algoritm
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    alpha: learning rate
    beta1: momentum weight
    var: numpy.ndarray containing variable to update
    grad: numpy.ndarray containing gradient of var
    v: previous first moment of var
    """
    V_t = beta1 * v + (1 - beta1) * grad
    W_var = var - alpha * V_t
    return W_var, V_t
