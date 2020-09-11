#!/usr/bin/env python3
"""
function performs forward propagation over
convolutional layer of NN
"""

import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """
    forward propagation with convolution layer
    """

    """
    A_prev: numpy.ndarray with shape(m, h_prev, w_prev, c_prev)
            contains output of previous layer
        m: num of examples
        h_prev: height of prev layer
        w_prev: width of prev layer
        c_prev: num channels in prev layer
    W: numpy.ndarray shape(kh, kw, c_prev, c_new)
       contains kernels for convolution
        kh: filter height
        kw: filter weight
        c_prev: num of channels in prev layer
        c_new: num channels in output
    b: numpy.darray shape(1, 1, 1, c_new)
       contains biases applied to convolution
    activation: activation function applied to convolution
    padding: string either 'same' or 'valid' type padding
    stride: tuple shape(sh, sw)
       sh: stride for height
       sw: stride for width
    """
    # retrive dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh = stride[0]
    sw = stride[1]

    # calculate padding for 'valid' and 'same'
    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2)
        ph = int(ph)
        pw = np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2)
        pw = int(pw)

    # compute output dimensions
    outputH = int(((h_prev - kh + (2 * ph)) / sh) + 1)
    outputW = int(((w_prev - kw + (2 * pw)) / sw) + 1)

    # initialize output with zeros
    outputFP = np.zeros((m, outputH, outputW, c_new))

    # Padding
    padImages = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                       mode='constant', constant_values=0)

    # vectorize
    padImagesVector = np.arange(0, m)

    # looping engine: vertical and horizontal
    for i in range(outputH):
        for j in range(outputW):
            for k in range(c_new):
                i_s = i * sh
                j_s = j * sw
                outputFP[padImagesVector, i, j, k] = activation((
                    np.sum(np.multiply(padImages[
                        padImagesVector,
                        i_s: i_s + kh, j_s: j_s + kw],
                                       W[:, :, :, k]),
                           axis=(1, 2, 3))) + b[0, 0, 0, k])
    return outputFP
