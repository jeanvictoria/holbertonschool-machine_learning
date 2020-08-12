#!/usr/bin/env python3
"""one_hot_decode"""

import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot is encoded numpy.ndarray w/ shape
    (classes, m)
    classes is the maximum number of classes
    m is the number of examples
    Returns: a numpy.ndarray with shape (m, ) containing the numeric label
      for each example, or None on failure"""

    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot.shape) != 2 or len(one_hot) == 0:
        return None
    if type(one_hot) is not np.ndarray:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    else:
        return np.argmax(one_hot, axis=0)
