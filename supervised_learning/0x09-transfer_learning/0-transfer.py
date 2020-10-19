#!/usr/bin/env python3
""" Transfer learning """
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    pre-processes the data for your model
    :param X: numpy.ndarray of shape (m, 32, 32, 3)
        containing the CIFAR 10 data,
        where m is the number of data points
    :param Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    :return: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_final = K.applications.inception_v3.preprocess_input(X)
    Y_final = K.utils.to_categorical(Y, 10)
    return X_final, Y_final


def model():
    """ build the model to classify the CIFAR 10
        Args:
            None
        Returns:
            the model.
    """
    inception = K.applications.InceptionV3(include_top=False,
                                           input_shape=(128, 128, 3))
    # for layer in inception.layers[:]:
    #     layer.trainable = False
    inception.layers.pop()

    model = K.Sequential()
    # model.add(K.layers.Lambda(lambda image:
    # K.backend.resize_images(image, (160, 160, 3))))
    model.add(K.layers.UpSampling2D(size=(4, 4)))
    model.add(inception)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(units=128,
                             activation='relu',
                             kernel_initializer='he_normal'))
    model.add(K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer='he_normal'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, alpha=0.1,
                decay_rate=1, filepath=None,
                verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent.
        Args:
            network: (tf.Tensor) model to train.
            data: (numpy.ndarray) containing the input data.
            labels: (numpy.ndarray) One-Hot array containing
                    the labels of data.
            batch_size: (int) the size of mini-batch.
            epochs: (int) number of epochs.
            verbose: (bool) determines if output should be
                     printed during training.
            shuffle: (bool) determines whether to shuffle the batches.
            validation_data: (tuple) containing the data to validate the model.
            early_stopping: (bool) incicates if early stopping should be used.
            patience: (int) patience using for early stopping.
            learning_rate_decay: (bool) indicates wheter learning rate decay
                                 should be used.
            alpha: (float) initial learning rate.
            decay_rate: (float) the decay rate.
            save_best: (bool) indicating whether to save the model
                       after each epoch.
            filepath: (str) path where the model should be saved.
        Returns:
            (tf.History) object generated after training the model.
    """
    def learning_rate_decay(epoch):
        """ learning rate callback """
        alpha_utd = alpha / (1 + (decay_rate * epoch))
        return alpha_utd

    callbacks = []

    checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                             save_best_only=True,
                                             monitor='val_loss',
                                             mode='min')
    callbacks.append(checkpoint)

    if validation_data:
        decay = K.callbacks.LearningRateScheduler(learning_rate_decay,
                                                  verbose=1)
        callbacks.append(decay)
    if validation_data and early_stopping:
        EarlyStopping = K.callbacks.EarlyStopping(patience=patience,
                                                  monitor='val_loss',
                                                  mode='min')
        callbacks.append(EarlyStopping)

    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)


if __name__ == '__main__':
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)
    model = model()
    train_model(model, X_train, Y_train, 64, 30,
                validation_data=(X_valid, Y_valid), early_stopping=True,
                patience=3, alpha=0.001, filepath='cifar10.h5')
