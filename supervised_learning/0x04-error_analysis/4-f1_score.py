#!/usr/bin/env python3
"""calculates the F1 score of a confusion matrix"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    classes is the number of classes
    """
    sen_arr = sensitivity(confusion)
    pre_arr = precision(confusion)
    F1_score = (2 * ((pre_arr * sen_arr) / (pre_arr + sen_arr)))

    return F1_score
