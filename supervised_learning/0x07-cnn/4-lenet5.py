#!/usr/bin/env python3
"""
modified version of LeNet-5 w/ tensorflow
"""

import tensorflow as tf


def lenet5(x, y):
    """
    x: placeholder shape (m, 28, 28, 1)
    y: placeholder shape (m, 10)
    m: num images
    model architecture:
        Convolutional layer: 6 kernels shape 5x5 same padding
        Max pooling layer: kernels of shape 2x2, 2x2 strides
        Convolutional layer: 16 kernels shape 5x5, valid padding
        Max pooling layer: kernels shape 2x2 with 2x2 strides
        Fully connected layer: 120 nodes
        Fully connected layer: 84 nodes
        Fully connected softmax output layer: 10 nodes
    Returns:
        a tensor for softmax activated output
        a training operation: Adam optimization, default hyperparameters
        a tensor: loss of the netowrk
        a tensor for: accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    act = tf.nn.relu

    layer1_conv = tf.layers.Conv2D(filters=6, kernel_size=5,
                                   padding='same', activation=act,
                                   kernel_initializer=init)(x)
    layer2_pool = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=2)(layer1_conv)
    layer3_conv = tf.layers.Conv2D(filters=16, kernel_size=5,
                                   padding='valid', activation=act,
                                   kernel_initializer=init)(layer2_pool)
    layer4_pool = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=2)(layer3_conv)

    """
    before moving to fully connected layers need to flatten outputs
    to reshape outputs
    """
    flatPool = tf.layers.Flatten()(layer4_pool)
    layer5_FC1 = tf.layers.Dense(units=120, activation=act,
                                 kernel_initializer=init)(flatPool)
    layer6_FC2 = tf.layers.Dense(units=84, activation=act,
                                 kernel_initializer=init)(layer5_FC1)
    layer7_output = tf.layers.Dense(units=10,
                                    kernel_initializer=init)(layer6_FC2)

    sMax_pred = tf.nn.softmax(layer7_output)
    loss = tf.losses.softmax_cross_entropy(y, layer7_output)
    trainOp = tf.train.AdamOptimizer().minimize(loss)
    equal = tf.equal(tf.argmax(y, axis=1), tf.argmax(layer7_output, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    return sMax_pred, trainOp, loss, accuracy
