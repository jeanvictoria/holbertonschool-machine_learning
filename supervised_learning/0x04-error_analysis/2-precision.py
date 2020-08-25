#!/usr/bin/env python3
"""calculates the precision for each class in a confusion matrix"""

import numpy as np


def precision(confusion):
    """
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    classes is the number of classes
    """
    precision = np.diagonal(confusion) / (np.sum(confusion, axis=0))
    return precision
