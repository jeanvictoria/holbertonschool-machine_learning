#!/usr/bin/env python3
"""Contains the l2_reg_gradient_descent function"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    pdates the weights and biases of a neural network using
        gradient descent with L2 regularization
    :param Y: one-hot numpy.ndarray of shape (classes, m)
        that contains the correct labels for the data
        classes is the number of classes
        m is the number of data points
    :param weights: a dict of the weights and biases of the neural network
    :param cache: dictionary of the outputs of each layer of the neural network
    :param alpha: learning rate
    :param lambtha: L2 regularization parameter
    :param L: number of layers of the network
    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation
    The weights and biases of the network should be updated in place
    """
    weights_t = weights.copy()
    m = Y.shape[1]

    for i in reversed(range(L)):
        if i == L - 1:
            dZ = cache['A{}'.format(i + 1)] - Y
            dW = (np.matmul(dZ, cache['A{}'.format(i)].T)) / m
        else:
            dZa = np.matmul(weights_t['W{}'.format(i + 2)].T, dZ)
            dZb = 1 - cache['A{}'.format(i + 1)]**2
            dZ = dZa * dZb
            dW = (np.matmul(dZ, cache['A{}'.format(i)].T)) / m

        dW_reg = dW + (lambtha / m) * weights_t['W{}'.format(i + 1)]
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W{}'.format(i + 1)] = weights_t['W{}'.format(i + 1)] \
            - (alpha * dW_reg)

        weights['b{}'.format(i + 1)] = weights_t['b{}'.format(i + 1)] \
            - (alpha * db)
