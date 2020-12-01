#!/usr/bin/env python3
"""Autoencoders"""

import tensorflow.keras as keras


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


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
    z_mean = keras.layers.Dense(latent_dims)(output)
    z_log_var = keras.layers.Dense(latent_dims)(output)
    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims, ))([z_mean, z_log_var])

    input_decoder = keras.Input(shape=(latent_dims,))
    out_decoder = keras.layers.Dense(hidden_layers[-1],
                                     activation='relu')(input_decoder)

    for layer in range(len(hidden_layers) - 2, -1, -1):
        out_decoder = keras.layers.Dense(hidden_layers[layer],
                                         activation='relu')(out_decoder)
    decoder_out = keras.layers.Dense(input_dims,
                                     activation='sigmoid')(out_decoder)

    encoder = keras.models.Model(inputs=input_image,
                                 outputs=[z, z_mean, z_log_var])
    decoder = keras.models.Model(inputs=input_decoder,
                                 outputs=decoder_out)

    full_encoder = encoder(input_image)[0]
    full_decoder = decoder(full_encoder)
    auto = keras.models.Model(inputs=input_image,
                              outputs=full_decoder)

    def loss(y_in, y_out):
        """ custom loss function """
        reconstruction_loss = keras.backend.binary_crossentropy(y_in, y_out)
        reconstruction_loss = keras.backend.sum(reconstruction_loss, axis=1)
        kl_loss = (1 + z_log_var - keras.backend.square(z_mean)
                   - keras.backend.exp(z_log_var))
        kl_loss = -0.5 * keras.backend.sum(kl_loss, axis=1)
        return reconstruction_loss + kl_loss

    auto.compile(optimizer='Adam',
                 loss=loss)

    return encoder, decoder, auto
