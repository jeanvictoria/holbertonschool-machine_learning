#!/usr/bin/env python3
"""
function builds a transition layer
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X: output from previous layer
    nb_filters: integer, representing num og filters in X
    compression: compression factor for transition layer
    implement compression as used in DenseNet-C
    weights: he_normal initialization
    Batch Norm layer-> Relu-> Conv layer
    Returns: output, num of filters w/in output
    """

    init = K.initializers.he_normal(seed=None)
    comp = int(compression * nb_filters)

    bNorm = K.layers.BatchNormalization()(X)
    act = K.layers.Activation('relu')(bNorm)
    conv1x1 = K.layers.Conv2D(filters=comp,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=init)(act)
    avgPool = K.layers.AveragePooling2D(pool_size=2)(conv1x1)
    return avgPool, comp
