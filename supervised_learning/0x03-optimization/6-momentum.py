#!/usr/bin/env python3
"""
function creates training op for NN in tf
uses gradient descent w/ momentum op
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    loss: loss of the network
    alpha: learning rate
    beta1: momentum weight
    """
    momentOpOp = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return momentOpOp
