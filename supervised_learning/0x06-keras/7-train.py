#!/usr/bin/env python3
"""
Update model
Train model with learning rate decay
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    learning_rate_decay: boolean indicateswether learning rate
                         decay is used
                         use: inverse time decay
                         decay: stepwise fashion after each epoch
                         rate updates: Keras print messag
    alpha: initial learning rate
    decay_rate: decay rate
    """

    def schedDecay(epoch):
        """inverse time decay learning rate"""
        return alpha * 1.0 / (1.0 + decay_rate * epoch)

    callBacks = []
    if validation_data and early_stopping:
        callBacks.append(K.callbacks.EarlyStopping(patience=patience,
                                                   monitor='val_loss'))

    if validation_data and learning_rate_decay:
        callBacks.append(K.callbacks.LearningRateScheduler(schedDecay, 1))

    history = network.fit(data, labels, epochs=epochs,
                          batch_size=batch_size, callbacks=callBacks,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle,)
    return history
