#!/usr/bin/env python3
"""2-size_me_please.py contain matrix_shape function"""


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    if matrix:
        shape = []
        while type(matrix) == list:
            shape.append(len(matrix))
            matrix = matrix[0]
        return shape
    else:
        return [0]
