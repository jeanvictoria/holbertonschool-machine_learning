#!/usr/bin/env python3
"""13-cats_got_your_tongue.py contain the np_cat function"""
import numpy as np


def np_cat(matrix1, matrix2, axis=0):
    """concatenates two matrices along a specific axis"""
    return np.concatenate((matrix1, matrix2), axis)
