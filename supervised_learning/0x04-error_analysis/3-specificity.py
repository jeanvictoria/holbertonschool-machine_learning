#!/usr/bin/env python3
"""calculates the specificity each class in a confusion matrix"""

import numpy as np


def specificity(confusion):
    """
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    classes is the number of classes
    """ 
    total = np.sum(confusion)
    truPos = np.diagonal(confusion)
    actual = np.sum(confusion, axis=1)
    predicted = np.sum(confusion, axis=0)
    falPos = predicted - truPos
    actNeg = total - actual
    truNeg = actNeg - falPos
    return truNeg / actNeg
