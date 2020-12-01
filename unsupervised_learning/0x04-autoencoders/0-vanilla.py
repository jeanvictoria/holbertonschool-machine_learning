#!/usr/bin/env python3
"""Autoencoders"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    :param input_dims:is an integer containing
    the dimensions of the model input
    :param hidden_layers: is a list containing the number
     of nodes for each hidden layer in the encoder, respectively
    :param latent_dims: is an integer containing the
     dimensions of the latent space representation
    :return:encoder, decoder, auto
    """
    input_image = keras.Input(shape=(input_dims,))
    output = keras.layers.Dense(hidden_layers[0],
                                activation='relu')(input_image)
    # encoder
    for layer in range(1, len(hidden_layers)):
        output = keras.layers.Dense(hidden_layers[layer],
                                    activation='relu')(input_image)
    encoder_out = keras.layers.Dense(latent_dims,
                                     activation='relu')(output)

    input_decoder = keras.Input(shape=(latent_dims,))
    out_decoder = keras.layers.Dense(hidden_layers[-1],
                                     activation='relu')(input_decoder)

    for layer in range(len(hidden_layers) - 2, -1, -1):
        out_decoder = keras.layers.Dense(hidden_layers[layer],
                                         activation='relu')(out_decoder)
    decoder_out = keras.layers.Dense(input_dims,
                                     activation='sigmoid')(out_decoder)

    encoder = keras.models.Model(inputs=input_image,
                                 outputs=encoder_out)
    decoder = keras.models.Model(inputs=input_decoder,
                                 outputs=decoder_out)

    input_auto = keras.Input(shape=(input_dims, ))

    full_encoder = encoder(input_auto)
    full_decoder = decoder(full_encoder)
    auto = keras.models.Model(inputs=input_auto,
                              outputs=full_decoder)
    auto.compile(optimizer='Adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
