#!/usr/bin/env python3
"""
function: saves model's weights
function: loads a model's weight
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves a model's weights
    """
    network.save_weights(filename.format(save_format))
    return None


def load_weights(network, filename):
    """
    loads saved weights
    """
    network.load_weights(filename)
    return None
