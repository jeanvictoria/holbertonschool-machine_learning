#!/usr/bin/env python3
"""
function performs a valid convolution
on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    images: numpy.ndarray w/ shape (m, h, w)
    kernel: numpy.ndarray w/ shape (kh, kw)
    Returns: a numpy.ndarray cotaining convolved images
    m: num of images
    h: height in pixels of images
    w: width in pixels of images
    kh: height of the kernel
    kw: width of the kernel
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    image = np.arange(0, m)
    cv_output = np.zeros((m, (h - kh + 1), (w - kw + 1)))
    for y in range(h - kh + 1):
        for x in range(w - kw + 1):
            cv_output[image, y, x] = (np.sum(images
                                      [image, y:kh + y, x:kw + x] *
                                      kernel,
                                      axis=(1, 2)))
    return cv_output
