#!/usr/bin/env python3
"""utilities module"""

import numpy as np
import glob
import cv2
import os


def load_images(images_path, as_array=True):
    """
    :param images_path: path to a directory from which to load images
    :param as_array: boolean indicating whether the images
        should be loaded as one numpy.ndarray
        If True, the images should be loaded as a numpy.ndarray
            of shape (m, h, w, c) where:
        m is the number of images
        h, w, and c are the height, width,
        and number of channels of all images, respectively
        If False, the images should be loaded as a list
        of individual numpy.ndarrays
    :return: images, filenames
        images is either a list/numpy.ndarray of all images
        filenames is a list of the filenames associated
        with each image in images
    """
    image_paths = glob.glob(images_path + '/*')
    image_paths = sorted([i for i in image_paths])

    filenames = [path.split('/')[-1] for path in image_paths]

    images_bgr = [cv2.imread(image) for image in image_paths]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]

    if as_array:
        images = np.array(images)

    return images, filenames


def load_csv(csv_path, params={}):
    """
    loads the contents of a csv file as a list of lists
    :param csv_path: path to the csv to load
    :param params: parameters to load the csv with
    :return: list of lists representing the contents found in csv_path
    """
    with open(csv_path, 'r') as f:
        csv_lines = [line.strip() for line in f]

    csv_list = [line.split(',') for line in csv_lines]

    return csv_list


def save_images(path, images, filenames):
    """
    saves images to a specific path
    :param path: path to the directory in which the images should be saved
    :param images: list/numpy.ndarray of images to save
    :param filenames: list of filenames of the images to save
    :return: True on success and False on failure
    """

    try:
        # Change the current directory
        os.chdir(path)

        for filename, image in zip(filenames, images):
            # Using cv2.imwrite() method to save the image
            cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Change back to working directory
        os.chdir('../')
        return True
    except FileNotFoundError:
        return False


def generate_triplets(images, filenames, triplet_names):
    """
    generates triplets
    :param images: numpy.ndarray of shape (n, h, w, 3)
        containing the various images in the dataset
    :param filenames: list of length n
        containing the corresponding filenames for images
    :param triplet_names: list of lists where each sublist contains
        the filenames of an anchor, positive, and negative image, respectively
    :return: [A, P, N]
        A is a numpy.ndarray of shape (m, h, w, 3)
            containing the anchor images for all m triplets
        P is a numpy.ndarray of shape (m, h, w, 3)
            containing the positive images for all m triplets
        N is a numpy.ndarray of shape (m, h, w, 3)
            containing the negative images for all m triplets
    """
    A = []
    P = []
    N = []

    stript_filenames = [filename.split('.')[0] for filename in filenames]

    for triplet in triplet_names:
        try:
            idx_A = stript_filenames.index(triplet[0])
            idx_P = stript_filenames.index(triplet[1])
            idx_N = stript_filenames.index(triplet[2])

            A.append(images[idx_A])
            P.append(images[idx_P])
            N.append(images[idx_N])
        except ValueError:
            pass

    return [np.array(A), np.array(P), np.array(N)]
