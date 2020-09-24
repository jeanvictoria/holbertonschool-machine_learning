#!/usr/bin/env python3
"""
function builds the DenseNet-121 architecture
"""

import tensorflow.keras as K


dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    growth_rate: growth rate
    compression: compression factor
    inputdata: shape (224, 224, 3)
    BatchNorm-> Relu-> conv layer
    weights: he_normal initialization
    Returns: keras model
    """

    inputData = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)
    bNorm = K.layers.BatchNormalization()(inputData)
    act = K.layers.Activation('relu')(bNorm)
    conv = K.layers.Conv2D(filters=64,
                           kernel_size=7,
                           strides=2,
                           padding='same',
                           kernel_initializer=init)(act)
    pool_1 = K.layers.MaxPooling2D(pool_size=3,
                                   strides=2,
                                   padding='same')(conv)
    output2, nbFilters2 = dense_block(pool_1, 64, growth_rate, 6)
    output3, nbFilters3 = transition_layer(output2, nbFilters2, compression)

    output4, nbFilters4 = dense_block(output3, nbFilters3, growth_rate, 12)
    output5, nbFilters5 = transition_layer(output4, nbFilters4, compression)

    output6, nbFilters6 = dense_block(output5, nbFilters5, growth_rate, 24)
    output7, nbFilters7 = transition_layer(output6, nbFilters6, compression)

    output8, nbFilters8 = dense_block(output7, nbFilters7, growth_rate, 16)

    avgPool = K.layers.AveragePooling2D(pool_size=7,
                                        strides=7,
                                        padding='same')(output8)
    outputs = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=init)(avgPool)
    return K.models.Model(inputData, outputs)
