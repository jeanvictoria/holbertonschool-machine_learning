#!/usr/bin/env python3
"""
Class Neuron that defines a single neuron
Forward propagation being calculated
"""
import numpy as np


class Neuron:
    """
    Neuron class
    """
    def __init__(self, nx):
        """
        Defines a single neuron performing binary classification
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weights"""
        return self.__W

    @property
    def b(self):
        """Bias"""
        return self.__b

    @property
    def A(self):
        """Actvation"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1/(1+np.exp(-z))
        return self.__A
