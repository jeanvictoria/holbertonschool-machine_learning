#!/usr/bin/env python3
"""
function creates a tensorflow layer include Dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    prev: tensor containing the output of the previous layer
    n: num of nodes the new layer should contain
    activation: activation function that should be used on the layer
    keep_prob: probability that a node will be kept
    """
    ker_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    newLayer = tf.layers.Dense(units=n, activation=activation,
                               kernel_initializer=ker_init)
    dropO = tf.layers.Dropout(keep_prob)
    return dropO(newLayer(prev))
