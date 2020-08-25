#!/usr/bin/env python3
"""calculates sensitivity for each class in a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """
    confusion: confusion numpy.ndarray of shape (classes, classes)
    classes: num of classes
    """
    sensitivity = np.diagonal(confusion) / (np.sum(confusion, axis=1))
    return sensitivity
