#!/usr/bin/env python3
"""Contains the model function"""

import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way
    :param X: first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    :param Y: second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y
    :return: the shuffled X and Y matrices
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm
    :param loss: loss of the network
    :param alpha: learning rate
    :param beta1: weight used for the first moment
    :param beta2: weight used for the second moment
    :param epsilon: small number to avoid division by zero
    :return: Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)


def create_layer(prev, n, activation):
    """
    create layer function
    :param prev: tensor output of the previous layer
    :param n: number of nodes in the layer to create
    :param activation: activation function that the layer should use
    :return: tensor output of the layer
    """
    # implement He et. al initialization for the layer weights
    initializer = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    model = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name='layer')

    return model(prev)


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow
    :param prev: activated output of the previous layer
    :param n: number of nodes in the layer to be created
    :param activation: activation function that should be used on
    the output of the layer
    :return:  tensor of the activated output for the layer
    """
    if activation is None:
        A = create_layer(prev, n, activation)
        return A

    # implement He et. al initialization for the layer weights
    initializer = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # dense layer model
    model = tf.layers.Dense(units=n,
                            kernel_initializer=initializer)
    Z = model(prev)

    # incorporation of trainable parameters beta and gamma
    # for scale and offset
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        name='gamma', trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       name='beta', trainable=True)
    epsilon = tf.constant(1e-8)

    # normalization parameter calculation
    mean, variance = tf.nn.moments(Z, axes=0)

    # Normalization over result after activation (with mean and variance)
    # and later adjusting with beta and gamma for
    # offset and scale respectively
    adjusted = tf.nn.batch_normalization(x=Z, mean=mean, variance=variance,
                                         offset=beta, scale=gamma,
                                         variance_epsilon=epsilon)

    A = activation(adjusted)
    return A


def forward_prop(x, layer, activations):
    """
    creates the forward propagation graph for the neural network
    :param x: placeholder for the input data
    :param layer: list containing the number of nodes in
        each layer of the network
    :param activations: list containing the activation functions
        for each layer of the network
    :return: prediction of the network in tensor form
    """
    # first layer activation with features x as input
    y_pred = create_batch_norm_layer(x, layer[0], activations[0])

    # successive layers activations with y_pred from the prev layer as input
    for i in range(1, len(layer)):
        y_pred = create_batch_norm_layer(y_pred, layer[i],
                                         activations[i])

    return y_pred


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction:
    :param y: placeholder for the labels of the input data
    :param y_pred: tensor containing the network’s predictions
    :return: tensor containing the decimal accuracy of the prediction
    """
    # from one y_pred one_hot to tag
    y_pred_t = tf.argmax(y_pred, 1)

    # from y one_hot to tag
    y_t = tf.argmax(y, 1)

    # comparison vector between tags (TRUE/FALSE)
    equal = tf.equal(y_pred_t, y_t)

    # average hits
    mean = tf.reduce_mean(tf.cast(equal, tf.float32))
    return mean


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction
    :param y: placeholder for the labels of the input data
    :param y_pred: tensor containing the network’s predictions
    :return: tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates a learning rate decay operation in tensorflow using
        inverse time decay:
    :param alpha: the original learning rate
    :param decay_rate: weight used to determine the rate at
        which alpha will decay
    :param global_step: number of passes of gradient descent that have elapsed
    :param decay_step: number of passes of gradient descent that should occur
        before alpha is decayed further
    :return: learning rate decay operation
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    Data_train is a tuple containing the training inputs and
               training labels, respectively
    Data_valid is a tuple containing the validation inputs and
               validation labels, respectively
    layers is a list containing the number of nodes in each
               layer of the network
    activation is a list containing the activation functions
               used for each layer of the network
    alpha is the learning rate
    beta1 is the weight for the first moment of Adam Optimization
    beta2 is the weight for the second moment of Adam Optimization
    epsilon is a small number used to avoid division by zero
    decay_rate is the decay rate for inverse time decay of
               the learning rate (the corresponding decay step should be 1)
    batch_size is the number of data points that should be in a mini-batch
    epochs is the number of times the training should pass
               through the whole dataset
    save_path is the path where the model should be saved to
    Returns: the path where the model was saved
    """
    # getting data_batch
    steps = Data_train[0].shape[0] / batch_size
    if (steps).is_integer() is True:
        steps = int(steps)
    else:
        steps = int(steps) + 1

    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    x = tf.placeholder(tf.float32, shape=[None, Data_train[0].shape[1]],
                       name='x')
    tf.add_to_collection('x', x)

    y = tf.placeholder(tf.float32, shape=[None, Data_train[1].shape[1]],
                       name='y')
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha,
                                decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    # initialize all variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

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
                # learning rate decay
                sess.run(global_step.assign(epoch))
                # update learning rate
                sess.run(alpha)

                # shuffle data, both training set and labels
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                # iteration within epoch
                for step_number in range(steps):

                    # data selection mini batch from training set and labels
                    start = step_number * batch_size

                    end = (step_number + 1) * batch_size
                    if end > Data_train[0].shape[0]:
                        end = Data_train[0].shape[0]

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
