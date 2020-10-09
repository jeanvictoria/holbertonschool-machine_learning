#!/usr/bin/env python3
"""Contains the train_mini_batch function"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
     trains a loaded neural network model using mini-batch gradient descent
    :param X_train: numpy.ndarray of shape (m, 784)
    containing the training data
        m is the number of data points
        784 is the number of input features
    :param Y_train: one-hot numpy.ndarray of shape (m, 10)
    containing the training labels
        10 is the number of classes the model should classify
    :param X_valid: numpy.ndarray of shape (m, 784)
    containing the validation data
    :param Y_valid: one-hot numpy.ndarray of shape (m, 10)
    containing the validation labels
    :param batch_size: number of data points in a batch
    :param epochs: number of times the training should pass
    through the whole dataset
    :param load_path: path from which to load the model
    :param save_path:  path to where the model
    should be saved after training
    :return: path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        # x is a placeholder for the input data
        x = tf.get_collection("x")[0]

        # y is a placeholder for the labels
        y = tf.get_collection("y")[0]

        # accuracy is an op to calculate the accuracy of the model
        accuracy = tf.get_collection("accuracy")[0]

        # loss is an op to calculate the cost of the model
        loss = tf.get_collection("loss")[0]

        # train_op perform one pass of gradient descent on the model
        train_op = tf.get_collection("train_op")[0]

        # find number of steps (iterations)
        steps = X_train.shape[0] // batch_size
        if steps % batch_size != 0:
            steps = steps + 1
            extra = True
        else:
            extra = False

        for epoch in range(epochs + 1):
            # execute cost and accuracy operations for training set
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})

            # execute cost and accuracy operations for validation set
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid})

            # where {epoch} is the current epoch
            print("After {} epochs:".format(epoch))

            # where {train_cost} is the cost of the model
            # on the entire training set
            print("\tTraining Cost: {}".format(train_cost))

            # where {train_accuracy} is the accuracy of the model
            # on the entire training set
            print("\tTraining Accuracy: {}".format(train_accuracy))

            # where {valid_cost} is the cost of the model
            # on the entire validation set
            print("\tValidation Cost: {}".format(valid_cost))

            # where {valid_accuracy} is the accuracy of the model
            # on the entire validation set
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:

                # shuffle data, both training set and labels
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                # iteration within epoch
                for step_number in range(steps):

                    # data selection mini batch from training set and labels
                    start = step_number * batch_size

                    if step_number == steps - 1 and extra:
                        end = X_train.shape[0]
                    else:
                        end = step_number * batch_size + batch_size

                    X = X_shuffled[start:end]
                    Y = Y_shuffled[start:end]

                    # execute training (from 0 to iteration) on mini set
                    sess.run(train_op, feed_dict={x: X, y: Y})

                    if step_number != 0 and (step_number + 1) % 100 == 0:
                        # where {step_number} is the number of times gradient
                        # descent has been run in the current epoch
                        print("\tStep {}:".format(step_number + 1))

                        # calculate cost and accuracy for mini set
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X, y: Y})

                        # where {step_cost} is the cost of the model
                        # on the current mini-batch
                        print("\t\tCost: {}".format(step_cost))

                        # where {step_accuracy} is the accuracy of the model
                        # on the current mini-batch
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
