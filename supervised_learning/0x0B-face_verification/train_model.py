#!/usr/bin/env python3
"""
class TrainModel
trains model for face verification using triplet loss
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from triplet_loss import TripletLoss


class TrainModel():
    """
    trains model for face verification using triplet loss
    """
    def __init__(self, model_path, alpha):
        """
        class constructor
        :model_path: path to base face verification embedding model
                    loads model using
                    with tf.keras.utils.CustomObjectScope({'tf': tf}):
                    saves model as public instance method base_model
        :alpha: use for the triplet loss calculation
                inputs: [A, P, N]
                A: numpy.ndarray contains anchor images
                P: numpy.ndarray contains pos images
                N: numpy.ndarray contains neg images
        :return: new model
        """
        # loads model and saves model
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)
        self.base_model.save(base_model)

        # inputs
        A_images = tf.placeholder(tf.float32, (None, 96, 96, 3))
        P_images = tf.placeholder(tf.float32, (None, 96, 96, 3))
        N_images = tf.placeholder(tf.float32, (None, 96, 96, 3))
        inputs = [A_images, P_images, N_images]

        # outputs encode
        output_images = self.base_model(inputs)

        # loss layer
        trip_loss = TripletLoss(alpha)
        outputs = trip_loss(output_images)

        # prepare training model
        training_model = K.models.Model(inputs, outputs)

        # compile using Adam
        training_model.compile(optimizer='Adam')

        # save training model
        training_model.save('training_model')

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        """
        create public instance method that trains
        self.training_model
        :param: triplets: list containing inputs to self.training_model
        :param: epochs: num of epochs to train for
        :param: batch_size: batch size for training
        :param: validation_split: validation split for training
        :param: verbose: boolean sets the verbosity mode
        Returns: History: output from training
        """
        history = self.training_mode.fit(
            triplets, validation_split=validation_split,
            batch_size=batch_size, epochs=epochs,
            verbose=verbose)
        return history

    def save(self, save_path):
        """
        create public instance method that saves
        the base embedding model
        :param save_path: path to save the model
        :return: saved model
        """
        self.base_model.save(save_path)

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        static method returns the f1 score
        :param y_pred: predictons
        :param y_true: actual
        :return: f1 score
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        static method returns the accuracy
        :param y_pred: prediction
        :param y_tru: actual
        :return:  accuracy
        """

        return K.metrics.BinaryAccuracy(y_true, y_pred)

    def best_tau(selfself, images, identities, thresholds):
        """
        public instance method
        :param images:
        :param identities:
        :param thresholds:
        :return: (tau, f1, acc)
        """
