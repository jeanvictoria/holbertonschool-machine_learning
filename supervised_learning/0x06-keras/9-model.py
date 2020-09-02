#!/usr/bin/env python3
"""
function save_model: saves entire model
function load_model: loadsd entire model
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    saves entire model
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    loads entire model
    """
    loaded_model = K.models.load_model(filename)
    return loaded_model
