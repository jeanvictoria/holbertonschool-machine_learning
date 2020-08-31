#!/usr/bin/env python3
"""
function creates a tensorflow layer include L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    prev: tensor containing the output of the previous layer
    n: num of nodes the new layer should contain
    activation: activation function that should be used on the layer
    lambtha: L2 regularization parameter
    """
    apply_L2Reg = tf.contrib.layers.l2_regularizer(lambtha)
    ker_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    newLayer = tf.layers.Dense(units=n, activation=activation,
                               kernel_initializer=ker_init,
                               kernel_regularizer=apply_L2Reg)
    return newLayer(prev)
