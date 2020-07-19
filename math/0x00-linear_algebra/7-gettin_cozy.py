#!/usr/bin/env python3
"""7-gettin_cozy.py"""


def cat_matrices2D(matrix1, matrix2, axis=0):
    """concatenates two matrices along a sepecific axis"""
    new_matrix = []
    if axis == 0:
        for i in matrix1 + matrix2:
            if len(i) != len(matrix1[0]):
                return None
            new_matrix.append(i[:])
    else:
        if len(matrix1) != len(matrix2):
            return None
        for j in range(0, len(matrix1), 1):
            new_matrix.append(matrix1[j] + matrix2[j])
    return new_matrix
