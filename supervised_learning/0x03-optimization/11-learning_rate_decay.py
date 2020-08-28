#!/usr/bin/env python3
"""
function updates learning rate using inverse time decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    alpha: original learning rate
    decay_rate: weight used to determine rate alpha will decay
    global_step: num passes gradient descent elapsed
    decay_step: num of passes gradient descent should occur
                before alpha is decayed further
    """
    return alpha / (1 + decay_rate * int(global_step / decay_step))
