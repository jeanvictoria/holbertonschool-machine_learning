#!/usr/bin/env python3
"""
function that builds the inception network
"""

import tensorflow.keras as K


inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Data in the shape(224, 224, 3)
    Convolutions inside & outside inception block
        use a rectified linear activation - ReLU
    Return: Keras Model
    """
    init = K.initializers.he_normal(seed=None)
    inputData = K.Input(shape=(224, 224, 3))

    netLayers = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                                padding='same', activation='relu',
                                kernel_initializer=init)(inputData)
    netLayers = K.layers.MaxPool2D(pool_size=3, strides=2,
                                   padding='same')(netLayers)

    netLayers = K.layers.Conv2D(filters=64, kernel_size=1,
                                padding='same', activation='relu',
                                kernel_initializer=init)(netLayers)
    netLayers = K.layers.Conv2D(filters=192, kernel_size=3,
                                padding='same', activation='relu',
                                kernel_initializer=init)(netLayers)
    netLayers = K.layers.MaxPool2D(pool_size=3, strides=2,
                                   padding='same')(netLayers)

    netLayers = inception_block(netLayers, [64, 96, 128, 16, 32, 32])
    netLayers = inception_block(netLayers, [128, 128, 192, 32, 96, 64])

    netLayers = K.layers.MaxPool2D(pool_size=3, strides=2,
                                   padding='same')(netLayers)

    netLayers = inception_block(netLayers, [192, 96, 208, 16, 48, 64])
    netLayers = inception_block(netLayers, [160, 112, 224, 24, 64, 64])
    netLayers = inception_block(netLayers, [128, 128, 256, 24, 64, 64])
    netLayers = inception_block(netLayers, [112, 144, 288, 32, 64, 64])
    netLayers = inception_block(netLayers, [256, 160, 320, 32, 128, 128])

    netLayers = K.layers.MaxPool2D(pool_size=3, strides=2,
                                   padding='same')(netLayers)

    netLayers = inception_block(netLayers, [256, 160, 320, 32, 128, 128])
    netLayers = inception_block(netLayers, [384, 192, 384, 48, 128, 128])

    avgPool = K.layers.AveragePooling2D(pool_size=7, strides=1)(netLayers)

    dropOut = K.layers.Dropout(.4)(avgPool)
    outData = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=init)(dropOut)
    return K.Model(inputData, outData)
