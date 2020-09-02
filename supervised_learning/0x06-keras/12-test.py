#!/usr/bin/env python3
"""
function tests a neural network
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    tests a NN
    """
    return network.evaluate(data, labels, verbose=verbose)
