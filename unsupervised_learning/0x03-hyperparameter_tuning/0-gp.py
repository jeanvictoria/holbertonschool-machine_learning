#!/usr/bin/env python3
"""
Class the represents a noisless 1D Gaussian Process
"""
import numpy as np


class GaussianProcess:

    """ Class reoresents noisless 1D Gaussian Process"""
    def __init__(self, X_init, Y_init, L=1, sigma_f=1):
        """
        class comstructor
        :param self:
        :param X_init: np.ndarray (t,1) imputs sampled with black-box function
        :param Y_init:np.ndarray (t, 1) outputs of the black box function
                        for X_init
        :param L: length parameter for the kernel
        :param sigma_f: standard deviation given output
        :return: covariance np.ndarray (m,n)
        """
        # t = 0
        # X_init = np.ndarray(shape=(t, 1))
        # Y_init = np.ndarray(shape=(t, 1))
        self.X = X_init
        self.Y = Y_init
        self.L = L
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two
        Matrices
        :param X1: np.ndarray, (m, 1)
        :param X2: np.ndarray, (n, 1)
        :return: covariance matrix (m, n)
        """
        # m = 0
        # n = 0
        # X1 = np.ndarray(shape=(m, 1))
        # X2 = np.ndarray(shape=(n, 1))

        # quared distance
        sqdist = np.sum(X1**2, 1).reshape(-1, 1)\
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        covMatrix = self.sigma_f**2 * np.exp(-0.5 / self.L**2 * sqdist)
        # print(covMatrix)
        return covMatrix
