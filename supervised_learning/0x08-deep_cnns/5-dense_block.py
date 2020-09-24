#!/usr/bin/env python3
"""
function builds dense block
"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    X: output from previous layer
    nb_filters: integer- num of filters
    growth_rate: growth rate for dense block
    layers: num of layers in dense block
    Use bottleneck layers for DenseNet-B
    weights use he_normal initialization
    conv layers preceded by batchNorm then ReLU layer
    Returns: concatenated output, num filters in outputs
    """

    init = K.initializers.he_normal(seed=None)

    for blocks in range(layers):
        bNorm_1 = K.layers.BatchNormalization()(X)
        act_1 = K.layers.Activation('relu')(bNorm_1)
        conv_1x1 = K.layers.Conv2D(filters=growth_rate * 4,
                                   kernel_size=1,
                                   padding='same',
                                   kernel_initializer=init)(act_1)
        bNorm_2 = K.layers.BatchNormalization()(conv_1x1)
        act_2 = K.layers.Activation('relu')(bNorm_2)
        # convolutional layer 3x3
        nextX = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=init)(act_2)
        X = K.layers.concatenate([X, nextX])
        nb_filters += growth_rate
    return X, nb_filters
