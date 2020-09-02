#!/usr/bin/env python3
"""
nx: num input features to NN
layers: list containing num of nodes in each layer of NN
activations: list containing action functions for layer
in NN
lambtha: L2 regularizer parameter
keep_prob: probability  that node will be kept for dropout
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds neural network with Keras using Input
    includes dropout
    """
    inputs = K.Input(shape=(nx,))
    b_reg = K.regularizers.l2(lambtha)
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=b_reg)(inputs)
    for lyr, act_f in zip(layers[1:], activations[1:]):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(lyr, activation=act_f,
                           kernel_regularizer=b_reg)(x)
    # instantiate the model given imputs and outputs
    model = K.Model(inputs=inputs, outputs=x)
    return model
