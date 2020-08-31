#!/usr/bin/env python3
"""
Function conducts forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    X: numpy.ndarray shape (nx, m) contains input data for NN
    weights: dict of weights & biases of NN
    L: number of layers in network
    keep_prob: probability that a node will be kept
    """
    cache = {}
    cache["A0"] = X
    for lyr in range(L):
        idx = (lyr + 1)
        Zprop = np.matmul(weights["W"+str(idx)], cache["A"+str(lyr)])
        Z = Zprop + weights["b"+str(idx)]
        drop_O = np.random.binomial(1, keep_prob, size=Z.shape)
        if lyr == L - 1:   # last layer
            cache["A"+str(idx)] = ((np.exp(Z)) / np.sum((np.exp(Z)),
                                                        axis=0, keepdims=True))
        else:  # updates
            cache["A"+str(idx)] = np.tanh(Z)   # layers use tanH activation
            cache["D"+str(idx)] = drop_O  # dropOut
            cache["A"+str(idx)] *= drop_O
            cache["A"+str(idx)] /= keep_prob
    return cache
