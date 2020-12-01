#!/usr/bin/env python3
"""Autoencoders"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters,
                latent_dims):
    """
    :param input_dims:is an integer containing
    the dimensions of the model input
    :param filters: is a list containing the number of
     filters for each convolutional layer in the encoder, respectively
    :param latent_dims: is an integer containing the
     dimensions of the latent space representation
    :return:encoder, decoder, auto
    """
    input_img = keras.Input(shape=input_dims)

    # encoder
    conv = keras.layers.Conv2D(filters=filters[0],
                               kernel_size=3,
                               activation='relu',
                               padding='same')(input_img)
    conv = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                     padding='same')(conv)

    for f in range(1, len(filters)):
        conv = keras.layers.Conv2D(filters=filters[f],
                                   kernel_size=3,
                                   activation='relu',
                                   padding='same')(conv)
        conv = keras.layers.MaxPool2D(pool_size=(2, 2),
                                      padding='same')(conv)

    output_encodder = conv

    # decoder

    input_decoder = keras.Input(shape=latent_dims)
    conv2 = keras.layers.Conv2D(filters=filters[-1],
                                kernel_size=3,
                                activation='relu',
                                padding='same')(input_decoder)
    conv2 = keras.layers.UpSampling2D(2)(conv2)

    for i in range(len(filters) - 2, 0, -1):
        conv2 = keras.layers.Conv2D(filters=filters[f],
                                    kernel_size=3,
                                    activation='relu',
                                    padding='same')(conv2)
        conv2 = keras.layers.UpSampling2D(2)(conv2)

    conv2 = keras.layers.Conv2D(filters=filters[0],
                                kernel_size=3,
                                padding='valid',
                                activation='relu')(conv2)
    conv2 = keras.layers.UpSampling2D(2)(conv2)

    output_decoder = keras.layers.Conv2D(filters=input_dims[-1],
                                         kernel_size=3,
                                         padding='same',
                                         activation='sigmoid')(conv2)

    encoder = keras.models.Model(inputs=input_img,
                                 outputs=output_encodder)
    decoder = keras.models.Model(inputs=input_decoder,
                                 outputs=output_decoder)

    input_auto = keras.Input(shape=input_dims)

    full_encoder = encoder(input_auto)
    full_decoder = decoder(full_encoder)
    auto = keras.models.Model(inputs=input_auto,
                              outputs=full_decoder)
    auto.compile(optimizer='Adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
