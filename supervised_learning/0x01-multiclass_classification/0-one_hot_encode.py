#!/usr/bin/env python3
"""one_hot_enconde"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Y is numpy.ndarray shape (m,) contains muneric class label
    m is num of example
    classes is the maximum number of classes found in
    Returns:a one-hot encoding of Y with shape (classes, m),
      or None on failure
    """
    if len(Y) == 0:
        return None
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int or classes <= np.max(Y):
        return None
    else:
        one_hot = np.zeros((classes, Y.shape[0]))
        for classes, m in enumerate(Y):
            one_hot[m][classes] = 1
        return one_hot
