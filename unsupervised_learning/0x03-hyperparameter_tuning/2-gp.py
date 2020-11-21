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
        print(covMatrix)
        return covMatrix

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points
                in a Gaussian process
        :param X_s: np.ndarray (s, 1)
        s: number of samplepoints
        mu: np.ndarray, (s,) contains mean for each point in X_s
        sigma: np.ndarray (s,) contains: standard deviation of each pointin X_s
        :return: mu, sigma
        """
        # s = 0
        # X_s = np.ndarray(shape=(s, 1))

        # update kernel witout adding sigma_y**2
        kern = self.kernel(self.X, self.X)
        kern_inv = np.linalg.inv(kern)
        kern_update = self.kernel(self.X, X_s)
        kern_update2 = self.kernel(X_s, X_s)

        # calculate the mean
        mean = kern_update.T.dot(kern_inv).dot(self.Y)
        mu = mean.reshape(-1)

        # calculate the sigma
        cov = kern_update2 - kern_update.T.dot(kern_inv).dot(kern_update)
        sigma = cov.diagonal()

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Update Public Instance Attributes X, Y, K
        :param X_new: np.ndarray (1,) new sample point
        :param Y_new: np.ndarray, (1,) new Sample function value
        :return:
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
