#!/usr/bin/env python3
"""
performs a convolution on images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same',
                      stride=(1, 1)):
    """
    images: numpy.ndarray w/ shape (m, h, w)
    kernel: numpy.ndarray w/ shape (kh, kw)
    padding: tuple of (ph, pw)
    Returns: a numpy.ndarray cotaining convolved images
    m: num of images
    h: height in pixels of images
    w: width in pixels of images
    c: num of channels in the image
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
    c = images.shape[3]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]

    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = int(((h-1)*sh+kh-h)/2) + 1
        pw = int(((w-1)*sw+kw-w)/2) + 1
    if type(padding) == tuple:
        ph = padding[0]
        pw = padding[1]
    if type(padding) == tuple or padding == 'same':
        images = np.pad(images,
                        pad_with=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    cust_sh = int(((h + 2*ph-kh) / sh) + 1)
    cust_sw = int(((w + 2*pw-kw) / sw) + 1)
    cv_output = np.zeros((m, cust_sh, cust_sw))
    image = np.arange(m)

    for y in range(cust_sh):
        for x in range(cust_sw):
            cv_output[image, y, x] = (np.sum(images
                                             [image, y*sh:(kh + (y*sh)),
                                              x*sw:(kw + (x*sw))] *
                                             kernel,
                                             axis=(1, 2, 3)))
    return cv_output
