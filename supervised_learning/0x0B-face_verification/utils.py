#!/usr/bin/env python3
"""
Face Verification
"""

import csv
import cv2
import numpy as np
import os
import glob


def load_images(images_path, as_array=True):
    """
    function loads images from directory or file
    image_path: path  to a directory  of images
    as_array: boolean indicating the images to be loaded
             images loaded as one numpy.ndarray
    Boolean factors: if true
                    images loaded as numpy.ndarray
                    shape(m, h, w, c)
                    if false
                    images loaded as list of indvidual
                    numpy.ndarray
        m: number of imagges
        h: height
        w: width
        c: num channels
    :return: images, filenames
        images: RGB format
                as list  of arrays or single array
        filenames: alphabetical Order
                as list of flienames assoc w/ each image
    """
    # image path
    imagePaths = glob.glob(images_path + "/*")

    # get image names
    imageNames = []
    for imPath in imagePaths:
        imageNames.append(imPaths.split('/')[-1])

    imageIndex = np.argsort(imageNames)

    # read and color images
    imageOrig = []
    for pics in imagePaths:
        imageOrig.append(cv2.imread(pics))
    for pics in imageOrig:
        imageOrig.append(cv2.cvtColor(pics, cv2.COLOR_BGR2RGB))

    images = []
    filenames = []

    # append images and file names
    for m in imageIndex:
        images.append(imageOrig[m])
        filenames.append(imageNames[m])

    # as_array
    if as_array:
        images = np.stack(images, axis=0)

    return images, filenames


def load_csv(csv_path, params={}):
    """
    :param csv_path: loads contents of csv files as list
    :param params: parameters to load csv with
    :return: list of lists rep contents found in csv_path
    """
    csvList = []
    with open(csv_path, 'r') as csvFile:
        csvReader = csv.reader(csvFile, params)
        for row in csvReader:
            csvList.append(row)
    return csvList


def save_images(path, images, filenames):
    """
    function that saves images images to a path
    :path: path to directory in which images should be saved
    :images: list/numpy.ndarray; images to save
    :filenames: list of filenames of the images to save
    :return: True on success
            False on failure
    """
    if os.path.exists(path):
        for imag, imagName in zip(images, filenames):
            pict = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
            cv2.imwrite('./' + path + '/' + imagName, pict)
        return True
    else:
        return False


def generate_triplets(images, filenames, triplet_names):
    """
    function that generates triplets
    :param images:
    :param filenames:
    :param triplet_names:
    :return: list [A, P, N]
    A is a numpy.ndarray shape (m, h, w, 3)
        containing anchor images for  m triplets
    P is a numpy.ndarray shape (m, h, w, 3)
        containing positive images for m triplets
    N is a numpy.ndarray shape (m, h, w, 3)
        containing negative images for m triplets
    """

    imagNames = [filenames[m].split('.')[0] for m in range(len(filenames))]

    ancNames = [names[0] for names in triplet_names]
    posNames = [names[1]for names in triplet_names]
    negNames = [names[2]for names in triplet_names]

    a_imgName = [m for m in range(len(imagNames)) if imagNames[m] in ancNames]
    p_imgName = [m for m in range(len(imagNames)) if imagNames[m] in posNames]
    n_imgName = [m for m in range(len(imagNames)) if imagNames[m] in negNames]

    A = images[a_imgName]
    P = images[p_imgName]
    N = images[n_imgName]

    print(A.shape)
    print(P.shape)
    print(N.shape)

    return [A, P, N]
