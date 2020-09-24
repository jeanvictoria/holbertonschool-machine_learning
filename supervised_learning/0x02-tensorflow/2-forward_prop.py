#!/usr/bin/env python3
"""function creates forward propagation graph for NN"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x is placeholder for input data
    layer_sizes is a list containing the number of nodes in each
       layer of the network
    activations is a list containing the activation functions for
       each layer of the networ
    Returns: the prediction of the network in tensor for
    """

    y_pred = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        y_pred = create_layer(y_pred, layer_sizes[i], activations[i])
    return y_pred
