#!/usr/bin/env python3
"""
performs pooling on images
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images: numpy.ndarray w/ shape (m, h, w)
    kernel: numpy.ndarray w/ shape (kh, kw)
    Returns: a numpy.ndarray cotaining convolved images
    m: num of images
    h: height in pixels of images
    w: width in pixels of images
    c: num channels in the image
    kh: height of the kernel
    kw: width of the kernel
    sh: stride height
    sw: stride width
    max: max pooling
    min: min pooling
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    height_sh = int(((h-kh)/sh) + 1)
    width_sw = int(((w-kw)/sw) + 1)
    image = np.arange(m)
    # ch_image = np.arange(c)
    cv_output = np.zeros((m, height_sh, width_sw, c))
    for y in range(height_sh):
        for x in range(width_sw):
            if mode == 'avg':
                cv_output[image, y, x] = (np.mean(images
                                          [image, y*sh:(kh + (y*sh)),
                                           x*sw:(kw + (x*sw))],
                                          axis=(1, 2)))
            if mode == 'max':
                cv_output[image, y, x] = (np.max(images
                                          [image, y*sh:(kh + (y*sh)),
                                           x*sw:(kw + (x*sw))],
                                          axis=(1, 2)))
    return cv_output
