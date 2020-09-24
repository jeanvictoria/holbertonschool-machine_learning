#!/usr/bin/env python3
"""
function build a projection block
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    A_prev: previous layer
    filters: is a tuple
    s: stride
    conv inside block followed by batch norm along
      channels axis
    weights use he_normal initialization
    Returns: activated output of projection block
    """

    F11, F3, F12 = filters

    init = K.initializers.he_normal(seed=None)

    out1 = K.layers.Conv2D(filters=F11, kernel_size=1, strides=s,
                           padding='same', kernel_initializer=init)(A_prev)
    bNorm_1 = K.layers.BatchNormalization()(out1)
    act_1 = K.layers.Activation('relu')(bNorm_1)

    out2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                           kernel_initializer=init)(act_1)
    bNorm_2 = K.layers.BatchNormalization()(out2)
    act_2 = K.layers.Activation('relu')(bNorm_2)

    out3 = K.layers.Conv2D(filters=F12, kernel_size=1,
                           padding='same',
                           kernel_initializer=init)(act_2)
    bNorm_3 = K.layers.BatchNormalization()(out3)

    o_short = K.layers.Conv2D(filters=F12, kernel_size=1,
                              padding='same', strides=s,
                              kernel_initializer=init)(A_prev)
    bNorm_short = K.layers.BatchNormalization()(o_short)

    addLayers = K.layers.Add()([bNorm_3, bNorm_short])
    return K.layers.Activation('relu')(addLayers)
