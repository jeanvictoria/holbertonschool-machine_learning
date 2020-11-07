#!/usr/bin/env python3
"""
Function calculates a GMM for a dataset
"""

import sklearn.mixture


def gmm(X, k):
    """
    GMM from data set
    :param X: np,ndarray, (n, d), data set
    :param k: num of cluaters
    pi: np.darray, (k, )
    m: np.ndarray, (k,d) centroid means
    S: np.ndarray, (k, d, d) covariance matrix
    clss: np.ndarray, (n,) cluster indices for data
    :return: pi, m, S, clss, bic
    """
    gMix = sklearn.mixture.GaussianMixture(k).fit(X)
    return (gMix.weights_, gMix.means_, gMix.covariances_,
            gMix.predict(X), gMix.bic(X))
