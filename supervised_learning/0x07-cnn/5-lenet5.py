#!/usr/bin/env python3
"""
modified version of LeNet-5 architechture using keras
"""

import tensorflow.keras as K


def lenet5(X):
    """
    X: a K.input shape (m, 28, 28, 1)
    m: num images
    Model architecture:
        Convolutional layer: 6 kernels shape 5x5 same padding
        Max pooling layer: kernels of shape 2x2, 2x2 strides
        Convolutional layer: 16 kernels shape 5x5, valid padding
        Max pooling layer: kernels shape 2x2 with 2x2 strides
        Fully connected layer: 120 nodes
        Fully connected layer: 84 nodes
        Fully connected softmax output layer: 10 nodes
    """
    init = K.initializers.he_normal(seed=None)
    act = 'relu'

    layer1_conv = K.layers.Conv2D(filters=6, kernel_size=5,
                                  padding='same', activation=act,
                                  kernel_initializer=init)(X)
    layer2_pool = K.layers.MaxPooling2D(pool_size=[2, 2],
                                        strides=2)(layer1_conv)
    layer3_conv = K.layers.Conv2D(filters=16, kernel_size=5,
                                  padding='valid', activation=act,
                                  kernel_initializer=init)(layer2_pool)
    layer4_pool = K.layers.MaxPooling2D(pool_size=[2, 2],
                                        strides=2)(layer3_conv)
    """
    before moving to fully connected layers need to flatten outputs
    to reshape outputs
    """
    flatPool = K.layers.Flatten()(layer4_pool)
    layer5_FC1 = K.layers.Dense(units=120, activation=act,
                                kernel_initializer=init)(flatPool)
    layer6_FC2 = K.layers.Dense(units=84, activation=act,
                                kernel_initializer=init)(layer5_FC1)
    layer7_output = K.layers.Dense(units=10, activation='softmax',
                                   kernel_initializer=init)(layer6_FC2)
    model = K.models.Model(X, layer7_output)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
