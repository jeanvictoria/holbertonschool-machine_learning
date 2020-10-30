#!/usr/bin/env python3
"""Multivariate Normal Distribution Class"""


import numpy as np


class MultiNormal:
    """Multivariate Normal Distribution Class"""
    def __init__(self, data):
        """
        class constuctor
        :param data: np.ndarray, shape(d, n)
        :param n: num data points
        :param d: num dimensions in each data point
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        """
        set the public instance variables
        """
        self.mean = ((np.mean(data, axis=1)).reshape(data.shape[0], 1))
        X = data - self.mean
        self.cov = ((np.dot(X, X.T)) / (data.shape[1] - 1))

    def pdf(self, x):
        """
        public instance method, calculates PDF of data point
        :param x: np.ndarray, shape(d, 1), contains datapoint PDF
                    to calculate
        :param d: num dimensions of Multinomial instance
        If x is not a numpy.ndarray,
        raise a TypeError with the message x must by a numpy.ndarray
        If x is not of shape (d, 1),
        raise a ValueError with the message x mush have the shape ({d}, 1)
        :return: value of PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if (len(x.shape) != 2):
            raise ValueError("x mush have the shape ({}, 1)".
                             format(self.cov.shape[0]))
        if (x.shape[1] != 1 or x.shape[0] != self.cov.shape[0]):
            raise ValueError("x mush have the shape ({}, 1)".
                             format(self.cov.shape[0]))
        # using det and inv of cov
            m = x - self.mean
            det = np.linalg.det(self.cov)
            pdf_det = 1. / (np.sqrt((2 * pn.pi)**self.cov.shape[0] * det))
            pdf_inv = (np.exp(-np.linalg.solve(self.cov, m).T.dot(m)) / 2)

            pdf = pdf_det * pdf_inv
            return pdf
