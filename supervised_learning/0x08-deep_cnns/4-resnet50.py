#!/usr/bin/env python3
"""function builds ResNet-50 architecture"""


import tensorflow.keras as K


identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    inputdata: shape (224, 224, 3)
    conv layer inside block follwed by batch norm layer
     along same channels
     and a ReLU
    weights use he_normal initialization
    Returns: keras model
    """

    init = K.initializers.he_normal(seed=None)
    iData = K.layers.Input(shape=(224, 224, 3))
    out1 = K.layers.Conv2D(filters=64,
                           kernel_size=7,
                           strides=2,
                           padding='same',
                           kernel_initializer=init)(iData)
    bNorm_1 = K.layers.BatchNormalization()(out1)
    act_1 = K.layers.Activation('relu')(bNorm_1)
    pool_1 = K.layers.MaxPooling2D(pool_size=3,
                                   strides=2,
                                   padding='same')(act_1)

    pout2 = projection_block(pool_1, [64, 64, 256], 1)
    out3 = identity_block(pout2, [64, 64, 256])
    out4 = identity_block(out3, [64, 64, 256])

    pout5 = projection_block(out4, [128, 128, 512])
    out6 = identity_block(pout5, [128, 128, 512])
    out7 = identity_block(out6, [128, 128, 512])
    out8 = identity_block(out7, [128, 128, 512])

    pout9 = projection_block(out8, [256, 256, 1024])
    out10 = identity_block(pout9, [256, 256, 1024])
    out11 = identity_block(out10, [256, 256, 1024])
    out12 = identity_block(out11, [256, 256, 1024])
    out13 = identity_block(out12, [256, 256, 1024])
    out14 = identity_block(out13, [256, 256, 1024])

    pout15 = projection_block(out14, [512, 512, 2048])
    out16 = identity_block(pout15, [512, 512, 2048])
    out17 = identity_block(out16, [512, 512, 2048])

    avgPool = K.layers.AveragePooling2D(pool_size=7,
                                        strides=None,
                                        padding='same')(out17)
    outputs = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=init)(avgPool)
    return K.models.Model(iData, outputs)
