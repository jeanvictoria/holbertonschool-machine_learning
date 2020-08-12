#!/usr/bin/env python3
"""
Contains the class DeepNeuralNetwork
"""
import numpy as np


class DeepNeuralNetwork:
    """class deepNeuralNetwork"""
    def __init__(self, nx, layers):
        """"constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if min(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {"W1": np.random.randn(layers[0], nx), "b1": np.zeros((layers[0], 1))}
        for i in range(1, self.L):
            self.weights[("W{}".format(i+1))] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1])
            self.weights[("b{}".format(i+1))] = np.zeros((layers[i], 1))
