#!/usr/bin/env python3
"""
function makes a prediction using a NN
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using NN
    """
    return network.predict(data, verbose=verbose)
