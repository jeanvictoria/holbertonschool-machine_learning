#!/usr/bin/env python3
"""
function builds a neural network w/ Keras
use dropout
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: num input features to the network
    layers: list containing num nodes in each layer of network
    activations: list contains activation functions for each layer NN
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout
    """

    b_reg = K.regularizers.l2
    model = K.Sequential([
        K.layers.Dense(layers[0], input_shape=(nx,), activation=activations[0],
                       kernel_regularizer=b_reg(lambtha))])
    for l, a in zip(layers[1:], activations[1:]):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(l, activation=a,
                                 kernel_regularizer=b_reg(lambtha)))
    return model
