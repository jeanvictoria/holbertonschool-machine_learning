#!/usr/bin/env python3
"""
Function performs K-means on a data
"""
import sklearn.cluster


def kmeans(X, k):
    """
    performs K-means on a dataset
    :param X: np.ndarray, (n,d), dataset
    :param k: num of clusters
    C: np.ndarray, shape(k,d), centroid means of each cluster
    clss: np.ndarray, shape(n,) index of cluxster in C
            for
    :return: C, clss
    """
    means = sklearn.cluster.KMeans(n_clusters=k).fit(X)

    return means.cluster_centers_, means.labels_
