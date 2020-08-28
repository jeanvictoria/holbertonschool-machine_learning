#!/usr/bin/env python3
"""function calculates weighed moving average of a data set"""

import numpy as np


def moving_average(data, beta):
    """
    data: list of data to calculate the moving average of
    beta: weight used for the moving average
    """
    ewa = 0
    bias_corrected = []
    for k in range(len(data)):
        ewa = (beta * ewa + (1 - beta) * data[k])
        bias_corrected.append(ewa / (1 - beta ** (k + 1)))
    return bias_corrected
