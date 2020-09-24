#!/usr/bin/env python3
"""
function builds inception block
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev: output from previous layer
    filters: tuple or list containing  F1, F3R,
        F3, F5R, F5, FPP
    F1: num filters in 1x1 convolution
    F3R: num filters in the 1x1 conv before 3x3 conv
    F3: num filters in 3x3 convolution
    F5R: num filters in 1x1 before the 5x5 conv
    F5: num filters in 5x5 convolution
    FPP: num filters in 1x1 conv after max pooling
    Convolution layers use a rectified linear activation (ReLu)
    Return: concatenated output of inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    con1x1 = K.layers.Conv2D(filters=F1, kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=init)(A_prev)
    con3x3 = K.layers.Conv2D(filters=F3R, kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=init)(A_prev)
    con3x3 = K.layers.Conv2D(filters=F3, kernel_size=3,
                             padding='same', activation='relu',
                             kernel_initializer=init)(con3x3)
    con5x5 = K.layers.Conv2D(filters=F5R, kernel_size=1,
                             padding='same', activation='relu')(A_prev)
    con5x5 = K.layers.Conv2D(filters=F5, kernel_size=5,
                             padding='same', activation='relu',
                             kernel_initializer=init)(con5x5)
    pool_ib = K.layers.MaxPool2D(pool_size=[3, 3], strides=1,
                                 padding='same')(A_prev)
    pool_ib = K.layers.Conv2D(filters=FPP, kernel_size=1,
                              padding='same', activation='relu',
                              kernel_initializer=init)(pool_ib)
    output = K.layers.concatenate([con1x1, con3x3, con5x5, pool_ib])
    return output
