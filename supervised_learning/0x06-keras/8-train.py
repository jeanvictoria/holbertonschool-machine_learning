#!/usr/bin/env python3
"""
updates training function
save the best iteration of the model
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    save_best: boolean indicates wether to save model after epoch
    filepath: file path where the model should be saved
    """

    def schedDecay(epoch):
        """inverse time decay learning rate"""
        return alpha * 1.0 / (1.0 + decay_rate * epoch)

    callBacks = []
    if validation_data:
        if early_stopping:
            callBacks.append(K.callbacks.EarlyStopping(patience=patience,
                                                       monitor='val_loss'))

        if learning_rate_decay:
            callBacks.append(K.callbacks.LearningRateScheduler(schedDecay, 1))

        if save_best:
            best = K.callbacks.ModelCheckpoint(filepath=filepath,
                                               monitor='val_loss',
                                               verbose=verbose,
                                               save_best_only=save_best,
                                               mode='min')
            callBacks.append(best)

    history = network.fit(data, labels, epochs=epochs,
                          batch_size=batch_size, callbacks=callBacks,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle)
    return history
