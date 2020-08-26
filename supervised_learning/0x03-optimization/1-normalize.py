#!/usr/bin/env python3
"""function normalizes(standardizes) a matrix"""

import numpy as np


def normalize(X, m, s):
    """
    X is numpy.ndarray of shape (d, nx) to normalize
    m is numpy.ndarray of shape (nx,) contains features of X
    s is numpy.ndarray of shape (nx,) contains Std Dev of X
    d id the num of data points
    nx is num of features
    """
    normX = (X - m) / s
    return (normX)
