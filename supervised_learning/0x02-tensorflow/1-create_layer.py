#!/usr/bin/env python3
"""function to create layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev is the tensor output of the previous laye
    n is the number of nodes in the layer to creat
    activation is the activation function that the layer should us
    """

    initializer = (tf.contrib.layers.
                   variance_scaling_initializer(mode="FAN_AVG"))
    return tf.layers.Dense(n, activation, name='layer',
                           kernel_initializer=initializer)(prev)
