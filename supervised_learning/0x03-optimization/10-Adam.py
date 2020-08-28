#!/usr/bin/env python3
"""
function creates training operation for NN in tf
using the Adam optimization algorithm
"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    loss: loss of the network
    alpha: learning rate
    beta1: weight used for first moment
    beta2: weight used for second moment
    epsilon: small number to avoid division by zero
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
