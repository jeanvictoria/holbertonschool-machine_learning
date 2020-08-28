#!/usr/bin/env python3
"""
normalizes an unactivated output of a neural network
using batch normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Z: numpy.ndarray shape (m, n) should be normalized
    gamma: numpy.ndarray shape (1, n) contain scales for batch norm
    beta: numpy.ndarray shape (1, n) contain offsetss for batch norm
    epsilon: small num used to avoid division by zero
    """

    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    std = np.sqrt(var + epsilon)
    Z_cent = (Z - mean)
    Z_norm = Z_cent / (std)
    Z_out = gamma * Z_norm + beta
    return Z_out
