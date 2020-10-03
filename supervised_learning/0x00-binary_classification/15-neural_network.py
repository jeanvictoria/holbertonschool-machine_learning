#!/usr/bin/env python3
"""Defines a neural network"""


import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """Neural Network Class"""
    def __init__(self, nx, nodes):
        """nx is input values; nodes are num nodes in hidden layer"""
        if type(nx) is not (int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not (int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """return weight vector for hidden layer"""
        return self.__W1

    @property
    def W2(self):
        """return weight vector output neuron"""
        return self.__W2

    @property
    def b1(self):
        """return bias for hidden layer"""
        return self.__b1

    @property
    def b2(self):
        """return bias for output neuron"""
        return self.__b2

    @property
    def A1(self):
        """return activated output hidden layer"""
        return self.__A1

    @property
    def A2(self):
        """return activated output for output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """calculates forward propagation of NN"""
        M = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1.0/(1.0 + np.exp(-M))
        self.__A2 = (np.dot(self.__W2, self.__A1) + self.__b2)
        self.__A2 = 1.0/(1.0 + np.exp(-1 * self.A2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """cost of model using logistic regression"""
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        """evaluate NN predictions"""
        return (self.forward_prop(X)[1].round().astype(int),
                self.cost(Y, self.__A2))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculates one pass gradient descent on the NN"""
        dz2 = A2 - Y
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        self.__W1 = self.__W1 - alpha * np.matmul(dz1, X.T) / X.shape[1]
        self.__W2 = self.__W2 - alpha * np.matmul(dz2, A1.T) / A1.shape[1]
        self.__b1 = self.__b1 - alpha * dz1.mean(axis=1, keepdims=True)
        self.__b2 = self.__b2 - alpha * dz2.mean(axis=1, keepdims=True)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """trains the neural network"""
        if type(iterations) is not (int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not (float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not (int):
                raise TypeError("step must be an integer")
            if step < 0 and step > iterations:
                raise ValueError("step must be positive and <= iterations")
        num_itr = 0
        num_cost = []
        num_pos = []
        A1, A2 = self.forward_prop(X)
        for num_itr in range(0, iterations):
            if verbose and (num_itr != step):
                print("Cost after {} iterations: {}"
                      .format(num_itr, self.cost(Y, A2)))
                if graph:
                    num_cost.append(self.cost(Y, A2))
                    num_pos.append(num_itr)
                self.gradient_descent(X, Y, A1, A2, alpha)
                num_itr += 1
            self.forward_prop(X)
            if verbose:
                print("cost after {} iterations: {}"
                      .format(num_itr, self.cost(Y, self.__A2)))
            if graph:
                plt.plot(num_pos, num_cost)
                plt.xlim(0, iterations)
                plt.xlabel("iteration")
                plt.ylabel("cost")
                plt.title("Training Cost")
            return self.evaluate(X, Y)
