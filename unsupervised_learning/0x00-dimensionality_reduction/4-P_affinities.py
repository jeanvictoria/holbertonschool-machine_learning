#!/usr/bin/env python3
"""
function calculates symmetric P affinities of a dataset
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    function calculates symmetric P affinities of a dataset
    :param X: np.ndarray, shape(n, d) contanins dataset to transform
    :param n: num of data points
    :param d: num of dimensions in each point
    :param tol: max tolerance allowed for difference in Shannon entropy
                from perplexity for all Gaussian distributions
    :param perplexity: perplexity all Gaussian distributions
    :return:
    """
    # variables
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)
    # logU = np.log(perplexity)
    # sumX = np.sum(np.square(X), 1)

    # loop over data points
    for idx in range(n):
        # Gaussian kernel and entropy
        betaMin = None
        betaMax = None
        # Di = D[i, np.concatenate((np.r_[0:idx], np.r_[i+1:n]))]
        Di = np.delete(D[idx], idx, axis=0)
        (Hi, Pi) = HP(Di, betas[idx])

        # calculate if perplexity is within tolerance
        diffH = Hi - H

        while np.abs(diffH) > tol:

            # adjust precision
            if diffH > 0:
                betaMin = betas[idx, 0]
                if betaMax is None:
                    betas[idx, 0] = beta[idx, 0] * 2
                else:
                    betas[idx, 0] = (betas[idx, 0] + betaMax) / 2
            else:
                betaMax = betas[idx, 0]
                if betaMin is None:
                    betas[idx] = betas[idx, 0] / 2
                else:
                    betas[idx, 0] = (betas[idx, 0] + betaMin) / 2

                (Hi, Pi) = HP(Di, betas[idx])
                diffH = Hi - H

        Pi = np.insert(Pi, idx, 0)
        P[idx] = Pi
        P = (P + P.T) / (2 * n)
        return (P)
