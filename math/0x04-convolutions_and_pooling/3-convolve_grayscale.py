#!/usr/bin/env python3
"""contains the convolve_grayscale function"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images
    :param images: numpy.ndarray with shape (m, h, w)
        containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    :param kernel: numpy.ndarray with shape (kh, kw)
        containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    :param padding:
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple: tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    :param stride: tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    :return: numpy.ndarray containing the convolved images
    """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]
    sw, sh = stride[1], stride[0]

    pw, ph = 0, 0

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if isinstance(padding, tuple):
        # Extract required padding
        ph = padding[0]
        pw = padding[1]

    # pad images
    images_padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    new_h = int(((images_padded.shape[1] - kh) / sh) + 1)
    new_w = int(((images_padded.shape[2] - kw) / sw) + 1)

    # initialize convolution output tensor
    output = np.zeros((m, new_h, new_w))

    # Loop over every pixel of the output
    for x in range(new_w):
        for y in range(new_h):
            # element-wise multiplication of the kernel and the image
            output[:, y, x] = \
                (kernel * images_padded[:,
                                        y * sh: y * sh + kh,
                                        x * sw: x * sw + kw]).sum(axis=(1, 2))

    return output
