#!/usr/bin/env python3
"""
performs a convolution on grayscale images
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    images: numpy.ndarray w/ shape (m, h, w)
    kernel: numpy.ndarray w/ shape (kh, kw)
    padding: tuple of (ph, pw)
    Returns: a numpy.ndarray cotaining convolved images
    m: num of images
    h: height in pixels of images
    w: width in pixels of images
    kh: height of the kernel
    kw: width of the kernel
    ph = padding for height
    pw = padding for width
    sh = stride for height
    sw = stride for width
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]
    # ph = padding[0]
    # pw = padding[1]
    output_h = 0
    output_w = 0

    if type(padding) == 'tuple':
        ph = padding[0]
        pw = padding[1]

    # valid has no padding
    if padding == 'valid':
        ph = 0
        pw = 0

    # if filter is odd or even
    if padding == 'same':
        ph = int(((h-1)*sh+kh-h)/2) + 1
        pw = int(((w-1)*sw+kw-w)/2) + 1
    if padding == 'same' or type(padding) == tuple:
        images = np.pad(images,
                        pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)

    cust_sh = int(((h - kh + (2*ph))/sh) + 1)
    cust_sw = int(((w - kw + (2*pw))/sw) + 1)
    cv_output = np.zeros((m, cust_sh, cust_sw))
    image = np.arange(0, m)
    for y in range(cust_sh):
        for x in range(cust_sw):
            cv_output[image, y, x] = (np.sum(images
                                      [image, y*sh:(kh + (y*sh)),
                                       x*sw:(kw + (x*sw))] *
                                      kernel,
                                      axis=(1, 2)))
    return cv_output
