#!/usr/bin/env python3
"""8-ridin_bareback.py contain the mat_mul fuction"""


def mat_mul(matrix1, matrix2):
    """performs matrix multiplication"""
    new_matrix = []
    for i in range(0, len(matrix1), 1):
        if len(matrix1[i]) != len(matrix2):
            return None
        row = []
        for j in range(len(matrix2[0])):
            sum = 0
            for i2 in range(len(matrix2)):
                sum += matrix1[i][i2] * matrix2[i2][j]
            row.append(sum)
        new_matrix.append(row)
    return new_matrix
