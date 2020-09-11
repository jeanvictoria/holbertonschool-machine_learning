#!/usr/bin/env python3
"""
performs forward Prop of Pooling layer
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    A-pre: np.ndarray containing output of prev layer
    m: num examples
    h_prev: height of previous layer
    w_prev: width previous layer
    c_prev: channels prev layer
    kernel_shape: tuple, size of kernel
    stride: tuple, stride for pooling
    mode: max or min
    """
    # retrive dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    # compute output dimensions
    outputH = int(((h_prev - kh) / sh) + 1)
    outputW = int(((w_prev - kw) / sw) + 1)

    # initialize output with zeros
    outputPFP = np.zeros((m, outputH, outputW, c_prev))

    # vectorize
    poImagesVector = np.arange(0, m)

    # max or min mode
    if (mode == 'avg'):
        activation = np.average
    if (mode == 'max'):
        activation = np.max

    # looping engine: vertical and horizontal
    for i in range(outputH):
        for j in range(outputW):
            i_s = i * sh
            j_s = j * sw
            outputPFP[poImagesVector, i, j] = activation(A_prev[
                poImagesVector, i_s: i_s + kh, j_s: j_s + kw],
                                                         axis=(1, 2))
    return outputPFP
