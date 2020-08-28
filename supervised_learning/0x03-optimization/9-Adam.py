#!/usr/bin/env python3
"""
function updates variable in place using Adam
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """
    alpha: learning rate
    beta1: weight used for first moment
    beta2: weight used for second moment
    epsilon: small number to avoid division by zero
    var: numpy.ndarray containing variable to update
    grad: numpy.ndarray containing gradient of var
    v: previous first moment of var
    s: previous second moment of var
    t: time step used for bias correction
    """

    """momentum for beta 1"""
    Vdv = beta1*v + (1 - beta1) * grad
    """RMSprop for beta 2"""
    Sdv = beta2*s + (1 - beta2) * grad**2
    """Corrected"""
    Vdv_corr = Vdv / ((1 - beta1**t))
    Sdv_corr = Sdv / ((1 - beta2**t))
    """updated variable"""
    W_var = var - alpha*(Vdv_corr / ((Sdv_corr**(1/2)) + epsilon))
    return W_var, Vdv, Sdv
