#!/usr/bin/env python3
"""
function trains a model using mini-batch gradient descent
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    network: model to train
    data: numpy.ndarray w/ shape (m, nx) contains input data
    labels: one-hot numpy.ndarray w/ shape (m, classes) cntains labels of data
    batch_size: size of the batch used for mini-batch grad descent
    epochs: num passes though data for mini-batch grad descent
    verbose: boolean determines if output is printed during training
    shuffle: boolean determines to shuffle batches every epoch
             set default to false
    Return: History object generated after  training the model
    """
    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          shuffle=shuffle, verbose=verbose)
    return history
