#!/usr/bin/env python3
"""
 Class Tripletloss inherits layer
"""
import tensorflow as tf
import tensorflow.keras as K


class Tripletloss:
    """
    inherits from tensorflow.keras.layers.Layer
    """

    def __init__(self, alpha, **kwargs):
        """
        :alpha: alpha value used to calculate the triplet loss
        """
        super(Tripletloss, self).__init__(**kwargs)
        self.alpha = alpha

    def triplet_loss(self, inputs):
        """
        calculates triplet loss
        :inputs: list containing: anchor, positive, negative output tensors
                from last layer of the model
        :return: tensor containing triplet loss value
        """

        A, P, N = inputs

        # anchor, positive image, negative image
        dist_1 = K.layers.Subtract()([A, P])
        dist_2 = K.layers.Subtract()([A, N])

        posDist = K.backend.sum(K.backend.square(dist_1), axis=1)
        negDist = K.backend.sum(K.backend.square(dist_2), axis=1)

        subLoss = K.layers.Subtract()([posDist, negDist]) + self.alpha
        loss = K.backend.maximum(subLoss, 0)

        return loss

    def call(self, inputs):
        """
        public instance method that calls Triplet Loss
        :inputs: list containning anchor, positive, negative output tensors
                from the last layer of the model
        :return: triplet tensor
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
