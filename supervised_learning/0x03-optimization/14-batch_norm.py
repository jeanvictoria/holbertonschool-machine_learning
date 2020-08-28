#!/usr/bin/env python3
"""
function creates  a batch normalization layer for NN
in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    prev: activated output of previous layer
    n: num of nodes in layer to be created
    activation: activation function used on output of layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    mm = tf.layers.Dense(units=n, activation=None,
                         kernel_initializer=init)
    batch_mean, batch_var = tf.nn.moments(mm(prev), axes=[0])

    beta = (tf.get_variable("beta", [n],
                            initializer=tf.zeros_initializer(),
                            trainable=True))
    gamma = (tf.get_variable("gamma", [n],
                             initializer=tf.ones_initializer(),
                             trainable=True))
    normlayer = (tf.nn.batch_normalization(mm(prev), mean=batch_mean,
                                           variance=batch_var, offset=beta,
                                           scale=gamma,
                                           variance_epsilon=1e-8))
    return activation(normlayer)
