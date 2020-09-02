#!/usr/bin/env python3
"""
functions: save and load configuration in json
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    save configuration in json format
    """
    json_network = network.to_json()
    with open(filename, 'w') as file:
        file.write(json_network)
    return None


def load_config(filename):
    """
    load json configuration
    """
    with open(filename, 'r') as jfile:
        json_network = K.models.model_from_json(jfile.read())
        return json_network
