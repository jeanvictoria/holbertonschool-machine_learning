#!/usr/bin/env python3
"""Contains the train function"""

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha, iterations,
          save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network classifier
    :param X_train: numpy.ndarray containing the training input data
    :param Y_train: numpy.ndarray containing the training labels
    :param X_valid: numpy.ndarray containing the validation input data
    :param Y_valid: numpy.ndarray containing the validation labels
    :param layer_sizes: list containing the number of nodes in
                        each layer of the network
    :param activations: list containing the activation functions
                        for each layer of the network
    :param alpha: learning rate
    :param iterations: number of iterations to train over
    :param save_path: designates where to save the model
    :return: the path where the model was saved
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    x, y = create_placeholders(nx, classes)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    # initialize all variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):

            # execute cost and accuracy operations for training set
            training_cost, training_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})

            # execute cost and accuracy operations for validation set
            validation_cost, validation_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(training_cost))
                print('\tTraining Accuracy: {}'.format(training_accuracy))
                print('\tValidation Cost: {}'.format(validation_cost))
                print('\tValidation Accuracy: {}'.format(validation_accuracy))

            # execute training (from 0 to iteration)
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        return saver.save(sess, save_path)
