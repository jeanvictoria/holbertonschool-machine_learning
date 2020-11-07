#!/usr/bin/env python3
"""
function performs agglomerative clustering on a dataset
"""


import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    agglomerative clustering on dataset
    :param X: np.ndarray, (X, dist) dataset
    :param dist: max cophentic dist for clusters
    clss: np.ndarray, (n, ) contains cluster indices for each data point
    :return: clss
    """
    h = scipy.cluster.hierarchy
    Z = h.linkage(X, 'ward')
    clust = h.fcluster(Z, t=dist, criterion="distance")
    fig = plt.figure()
    dn = h.dendrogram(Z, color_threshold=dist)
    plt.show()
    return clust
