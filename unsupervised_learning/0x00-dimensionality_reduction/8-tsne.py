#!/usr/bin/env python3
"""
performs a t-SNE transformation
"""

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    function performs a t-SNE transformation
    :param X: np.ndarray, shape(n,d) contains dataset
    :param ndims: new dimensioinal representation of X
    :param idims: intermediate dim rep of X after PCA
    :param perplexity: the perplexity
    :param iterations: num of iterations
    :param lr: learning rate
    :return: Y, np.ndarray, shape(n, ndim)
            containing optimized low dimensional transformation of X
    """
    # variables
    (n, d) = X.shape
    X = pca(X)
    at_i = 0.5
    at_f = 0.8
    Y = np.random.randn(n, ndims)
    dY = np.zeros((n, ndims))
    iY = np.zeros((n, ndims))
    updaete = np.zeros((n, ndims))
    update_min = 0.01

    # P values
    P = P_affinities(X,  le-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4
    P = np.maximum(P, 1e-12)

    # iteratinos
    for iters in range(interations):

        # Q values
        Ysum = np.sum(np.square(Y), 1)
        Qnum = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), Ysum).T, Ysum))
        Qnum[range(n), range(n)] = 0
        Q = Qnum / np.sum(Qnum)
        Q = np.maximum(Q, 1e-12)

        # gradient
        # diff_PQ = P - Q
        for i in range(n):
            grads(Y, P)

        if iters < 20:
            momentum = at_i
        else:
            momentum = at_f
        update = (update + 0.2) * ((dY > 0) != (iY > 0)) +\
                 (update * 0.8) * ((dY > 0) == (iY > 0))
        update[update < update_min] = update_min
        iY = momentum * iY - lr * (update * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Cost
        if (iters + 1) % 10 == 0:
            cost(X)

        if iters == 100:
            P = P / 4
    return Y
