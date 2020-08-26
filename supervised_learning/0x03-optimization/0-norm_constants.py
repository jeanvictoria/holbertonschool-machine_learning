#!/usr/bin/env python3
"""function calculates normalization constants of a matrix"""


import numpy as np


def normalization_constants(X):
    """calcuate normalization"""
    return X.mean(axis=0), X.std(axis=0)
