#!/usr/bin/env python3
"""
Class NeuralNetwork
Defines a neural network with one hidden
"""
import numpy as np


class NeuralNetwork:
    """Class Neural Network"""
    def __init__(self, nx, nodes):
        """constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weights for hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Bisas for hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Activated output for hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Weights for output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Activated output for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1+np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1+np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        cost = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.A2)
        pred_labels = np.where(self.A2 >= 0.5, 1, 0)
        return pred_labels, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(A1, dZ2.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(X, dZ1.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        self.__W1 = self.__W1 - (alpha * dW1).T
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dW2).T
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        return self.evaluate(X, Y)
