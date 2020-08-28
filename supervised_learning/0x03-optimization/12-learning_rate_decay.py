#!/usr/bin/env python3
"""
function creates learning rate operation in tf
using inverse time decay
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    alpha: original learning rate
    decay_rate: weight used to determine rate at alpha will decay
    global_step: num passes of gradient descent elapsed
    decay_step: num passes gradient descent that occur
                before alpha is decayed further
    """
    return (tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True))
