#!/usr/bin/env python3
"""
function performs back propagatino over pooling layer
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    dZ: np.ndarray, partial derivatives
        m: num examples
        h_new: output height
        w_new: output weight
        c_new: output channels
    A_prev: np.ndarray, prev output layer
        m: num examples
        h_prev: prev height
        w_prev: prev width
        c_prev: prev channels
    kernel_shape: tuple, (kh, kw)
    stride: tuple, (sh, sw)
    mode: max or min
    """
    # retrive dimensions
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    # initializing output: dA_prev, dW, db
    dA_prev = np.zeros(A_prev.shape)

    # vectorize
    poImagesVector = np.arange(0, m)

    # max or min mode
    if (mode == 'avg'):
        activation = np.average
    if (mode == 'max'):
        activation = np.max

    # Looping engine: loop over m, h_new, w_new, c_new
    for n in range(m):
        a_prev = A_prev[n, :, :, :]

        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    # finding the corners
                    istart = i * sh
                    iend = istart + kh
                    jstart = j * sw
                    jend = jstart + kw
                    dA_prev[poImagesVector, i, j, k] = activation(
                        (dA_prev[n, i, j, k] + dA[n, i, j, k]))
    return dA_prev
