#!/usr/bin/env python3
"""5-across_the_planes.py contain add_matrices2D function"""


def add_matrices2D(matrix1, matrix2):
    """add two matrices element-wise"""
    if len(matrix1) != len(matrix2):
        return None
    new_matrix = []
    for i in range(len(matrix1)):
        if len(matrix1[i]) != len(matrix2[i]):
            return None
        row = []
        for j in range(len(matrix1[i])):
            row.append(matrix1[i][j] + matrix2[i][j])
        new_matrix.append(row)
    return new_matrix
