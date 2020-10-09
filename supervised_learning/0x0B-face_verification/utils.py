#!/usr/bin/env python3
""" Face Varification System """

import numpy as np
import os
import glob
import cv2
import csv


def load_images(images_path, as_array=True):
    """ Loading the images in a RGB format from a directory or file by
        alphabetical order  by filename.
    Arg:
        images_path: path to a directory from which to load images
        as_array: boolean indicating the images should be
                  loaded as one numpy.ndarray or not
               If True, load the imgs numpy.ndarray of shape (m, h, w, c)
               If False, load the imgs as a list of individual np.ndarrays
    Returns: images, filenames
        images: is either a list/numpy.ndarray of all images
        filenames: is a list of the filenames of each image
    """
    if os.path.exists(images_path):
        if os.path.isdir(images_path):
            # if images_path is a directory, else if it is a filename
            path = images_path + '/*.jpg'
            image_paths = glob.glob(path, recursive=False)
            image_paths.sort()
        else:
            image_paths = images_path

    loaded_images = []
    images_names = []

    for i, img in enumerate(image_paths):
        src = cv2.imread(img)
        # convert the image into a RGB format
        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        # extracting the image name of the file_path
        name = image_paths[i].split('/')[-1]
        loaded_images.append(image)
        images_names.append(name)

    if as_array:
        loaded_images = np.stack(loaded_images, axis=0)

    return (loaded_images, images_names)


def load_csv(csv_path, params={}):
    """ loads the content of a CVS file a lists of lists
    Arg:
        csv_path is the path to the csv to load
        params are the parameters to load the csv with
    Returns: list of lists representing the contents found in csv_path
    """
    csv_content = []

    with open(csv_path, encoding="utf-8") as csv_file:
        # used csv (comma separated values file)
        csv_reader = csv.reader(csv_file, params)
        for line in csv_reader:
            csv_content.append(line)
    return csv_content


def save_images(path, images, filenames):
    """saves images to a specific path
    Arg:
        path: path to the directory in which the images should be saved
        images: list/numpy.ndarray of images to save
        filenames: list of filenames of the images to save
    Returns: True on success and False on failure
    """
    if os.path.exists(path):
        for img, name in zip(images, filenames):
            # convert the image into a RGB format
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('./' + path + '/' + name, image)
        return True
    else:
        return False


def generate_triplets(images, filenames, triplet_names):
    """ generates the triplets
    Arg:
        images: np.ndarray shape (n, h, w, 3) with images in the dataset
        filenames: list leng n with the corresponding filenames for images
        triplet_names: list of lists where each sublist contains the
                 filenames of an anchor, positive, and negative image
    Returns: a list [A, P, N]
       - A np.ndarray shape (m, h, w, 3) has anchor images for all m triplets
       - P np.ndarray (m, h, w, 3) has the positive images for all m triplets
       - N np.ndarray (m, h, w, 3) has the negative images for all m triplets
    """
    inds_a, inds_p, inds_n = [], [], []

    names = [filenames[i].split('.')[0] for i in range(len(filenames))]
    # Reeplace values with differents characters
    for triplet_name in triplet_names:
        for name in triplet_name:
            if name not in names:
                # Reeplace special characters as í, é, ñ
                new_value = name.encode('utf-8').decode('utf-8')
                new_value = new_value.replace('eÌ', 'é')
                new_value = new_value.replace('iÌ', chr(105) + chr(769))
                new_value = new_value.replace('nÌƒ', chr(110)+chr(771))
                new_value = new_value.replace('\x81', '')
                new_value = new_value.replace('\x81', '')
                # Find the name of the triplet in names array and replace it
                triplet_name[triplet_name.index(name)] = new_value

    # Create the list with indices of the values
    for triplet_name in triplet_names:
        inds_a.append(names.index(triplet_name[0]))
        inds_p.append(names.index(triplet_name[1]))
        inds_n.append(names.index(triplet_name[2]))

    # Create arrays with the images
    A = images[inds_a]
    P = images[inds_p]
    N = images[inds_n]

    return [A, P, N]
