#!/usr/bin/env python3
"""calculate_accuracy"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """

    prediction = tf.argmax(y_pred, 1)
    results = tf.argmax(y, 1)
    equal = tf.equal(prediction, results)
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    return accuracy
