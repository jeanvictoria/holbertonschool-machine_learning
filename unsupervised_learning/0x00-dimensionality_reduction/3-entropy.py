#!/usr/bin/env python3
"""
calculates the Shannon entropy and P affinities relative to a data point
"""
import numpy as np


def HP(Di, beta):
    """
    calculates the Shannon entropy and P affinities relative to a data point
    :param Di: Di is a numpy.ndarray of shape (n - 1,)
    Hi: the Shannon entropy of the points
    Pi: a numpy.ndarray of shape (n - 1,)
    containing the P affinities of the points
    :param beta: beta is the beta value for the Gaussian distribution
    :return:
    """

    Pi = (np.exp(-Di * beta)) / (np.sum(np.exp(-Di * beta)))
    Hi = - np.sum(Pi * np.log2(Pi))

    return (Hi, Pi)
