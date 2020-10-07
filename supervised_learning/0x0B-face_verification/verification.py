#!/usr/bin/env python3
"""
Face Verification Class
"""
import tensorflow as tf


class FaceVerification:
    """
    Face Verification Class
    """
    def __init__(self, model, database, identities):
        """
        class constructor
        :param self:
        :param model: face verification embedding model
                or path to where the model is stored
        :param database: numpy.ndarray: contains all face
                    embeddings in database
        :param identities: list of identities corresponding to
                        embeddings in database
        :public instance attributes:  database, identities
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = tf.keras.models.load_model(model)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        """
        embedding images
        :param self:
        :param images: images to retrieve embeddings of
        :return: numpy.ndarray of embeddings
        """
        embeddings = self.model.predict(images)
        return embeddings

    def verify(self, image, tau=0.5):
        """
        verify image identity
        :param self: Any
        :param image: aligned image of face to verify
        :param tau: float = 0.5 Union[tulpe[any,any], Tuple[None, None]]
                max euclidean distance used to verify
        :return param: identity : identity of the verified face
        :return param: distence: euclidean distance bewteeen the
                        verified face embedding
                        and identified database embedding
        :return: (identity, distance) or (None, None)
        """
        a = (self.identities - self.database)
        distance = np.sum(np.square(tf.broadcast_to.embeddings(a)))
        if (distance > tau):
            return(identity, distance)
        else:
            return (None, None)
