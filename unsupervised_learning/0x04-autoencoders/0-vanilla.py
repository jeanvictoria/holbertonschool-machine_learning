#!/usr/bin/python3
"""
A function that creates an autoencoder
"""

import tensorflow as tf
from tensorflow import keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates an autoencoder
    :param input_dims: int contains dimensions of model input
    :param hidden_layers: list, num of nodes for each layer in encoder
    :param latent_dims: int, dimensions of the latent space
                        representation
    note: hiden layers should be reversed for the decoder
    note: autoencoder model should be compiled using adam
    note: all layers use a relu activation except the last layer
    note: last lahyer in the decoder uses sigmoid
            encoder: encoder model
            decoder: decoder model
            auto: full autoencoder model
    :return:  encoder, decoder, auto
    """
    # inputs
    encoderInput = K.Input(shape=(input_dims, ))
    decoderInput = K.Input(shape=(latent_dims, ))

    # first output layers
    en_output_1 = K.layers.Dense(hidden_layers[0],
                                 activation='relu')(encoderInput)
    de_output_1 = K.layers.Dense(hidden_layers[-1],
                                 activation='relu')(decoderInput)

    # Encoder
    for idx in range(1, len(hidden_layers)):
        # second output layer
        en_output_2 = K.layers.Dense(hidden_layers[idx],
                                     activation='relu')(en_output_1)
    encoderOutput = K.layers.Dense(latent_dims,
                                   activation='relu')(en_output_2)

    # encoder Model
    encoder = K.models.Model(inputs=encoderInput, outputs=encoderOutput)

    # Decoder
    for idx in range(len(hidden_layers)-2, -1, -1):
        # second output layer
        de_output_2 = K.layers.Dense(hidden_layers[idx],
                                     activation='relu')(de_output_1)
    decoderOutput = K.layers.Dense(latent_dims,
                                   activation='sigmoid')(de_output_2)

    # decoder Model
    decoder = K.models.Model(inputs=decoderInput, outputs=decoderOutput)

    # Autoencoder
    autoInput = K.Input(shape=input_dims, )
    encoderAuto = encoder(autoInput)
    decoderAuto = decoder(encoderAuto)
    auto = K.Model(inputs=autoInput, outputs=decoderAuto)
    auto.K.model.compile(optimizer='Adam', loss='binary_crossentropy')

    return (encoder, decoder, auto)
