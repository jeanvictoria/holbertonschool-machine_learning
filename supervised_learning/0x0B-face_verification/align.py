#!/usr/bin/env python3
"""
Face Align Class
"""
import numpy as np
import cv2
import dlib


class FaceAlign:
    """
    Face Align Class
    """
    def __init__(self, shape_predictor_path):
        """
        :param self:
        :shape_predictor_path: path to dlib shape predictor model
        :detector: contains dlib's default face detector
        :shape_predictor: contains the dlib.shape_predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        :image: numpy.ndarray rank 3 contains image from which to detect face
        :dlib.rectangle: contains boundary box for face in image
                     or None on failure
                    -if multiple faces detected return dlib.rectangle
                    with the largest area
                    -if no faces are detected return dlib.rectangle
                    the same as the image
        :return: dlib.retangle
        """
        try:
            faces = self.detector(image, 1)
            area = 0

            for aface in faces:
                if aface.area() > area:
                    area = aface.area()
                    rect = aface

            if area == 0:
                rect = (dlib.rectangle(left=0,
                                       top=0,
                                       right=image.shape[1],
                                       botom=image.shape[0]))
            return rect
        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """
        :image: a numpy.darray of an image from which to find
                facial
        :detection: dlib.rectangle containing boundary box of the
        :return: numpy.darray shape(p, 2)
        """
        landmarkPoints = self.shape_predictor(image, detection)
        if not landmarkPoints:
            return None

        coordpts = np.zeros((68, 2), dtype='int')
        for m in range(0, 68):
            coordpts[m] = [landmarkPoints.part(m).x, landmarkPoints.part(m).y]
        return coordpts

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        :image: a numpy.ndarray rank 3 containing image to be aligned
        :landmark_indices: numpy.ndarray shape(3, )
                            containing indices of 3 landmark points
                            to be used for affine transformation
        :anchor_points: numpy.ndarray shape(3,2) containing destination points
                        for affine transformation
                        scaled to the range[0, 1]
        :size: desired size of aligned image
        :return: numpy.darray shape(size, size, 3)
                 contains aligned image
                 if no face detected contains None
        """
        rectBox = self.detect(image)
        coordPts = self.find_landmarks(image, rectBox)
        coordPts = coordPts.astype('float32')
        faceAnchors = anchor_points * size
        warp_mat = cv2.getAffineTransform(coordPts, faceAnchors)
        warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

        return warp_dst
