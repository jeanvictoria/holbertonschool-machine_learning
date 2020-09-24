#!/usr/bin/env python3
"""
function builds identity block
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    function builds identity block
    filters: is tuple
    convolution inside block followed by batch norm along
       channels axis
    weights use he_normal initilization
    Returns: activated outpit of identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    out1 = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                           kernel_initializer=init)(A_prev)
    bNorm_1 = K.layers.BatchNormalization()(out1)
    act_1 = K.layers.Activation('relu')(bNorm_1)

    out2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                           kernel_initializer=init)(act_1)
    bNorm_2 = K.layers.BatchNormalization()(out2)
    act_2 = K.layers.Activation('relu')(bNorm_2)

    out3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                           kernel_initializer=init)(act_2)
    bNorm_3 = K.layers.BatchNormalization()(out3)

    addLayers = K.layers.Add()([bNorm_3, A_prev])
    return K.layers.Activation('relu')(addLayers)
