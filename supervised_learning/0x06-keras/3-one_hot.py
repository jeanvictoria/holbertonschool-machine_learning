#!/usr/bin/env python3
"""
function converts a lable vector into a one-hot matrix
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    convert vector to one-hot matrix
    """
    encoded = K.utils.to_categorical(labels, classes)
    return(encoded)
