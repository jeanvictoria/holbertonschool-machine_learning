#!/usr/bin/env python3
"""
function to analyze validation data
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    validation_data is the data to validate the model
    if not None
    """

    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle,)
    return history
