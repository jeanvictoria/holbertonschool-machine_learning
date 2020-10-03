#!/usr/bin/env python3
"""deep NN performing binary classififcation"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network Class"""

    def __init__(self, nx, layers):
        """nx is number of input values"""
        if type(nx) is not (int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        """layers list reping num nodes in each layer"""
        if type(layers) is not (list) or len(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.nx = nx
        self.cache = {}
        self.weights = {}
        for i_lyr in range(self.L):
            mWts = "W" + str(i_lyr + 1)
            mB = "b" + str(i_lyr + 1)
            if type(layers[i_lyr]) is not (int) or layers[i_lyr] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights[mB] = np.zeros((layers[i_lyr], 1))
            if i_lyr == 0:
                self.weights[mWts] = (np.random.randn(layers[i_lyr], nx)
                                      * np.sqrt(2 / nx))
            else:
                self.weights[mWts] = (np.random.randn(layers[i_lyr],
                                      layers[i_lyr - 1])
                                      * np.sqrt(2 / layers[i_lyr - 1]))
