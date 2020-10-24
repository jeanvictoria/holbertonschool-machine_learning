#!/usr/bin/env python3
"""contain the minor method"""


def determinant(matrix):
    """
    calculates the determinant of a matrix:
    :param matrix: list of lists
        whose determinant should be calculated
    :return: determinant of matrix
    """

    err = 'matrix must be a list of lists'
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError(err)

    my_len = len(matrix)
    if my_len == 1 and len(matrix[0]) == 0:
        return 1

    for element in matrix:
        if not isinstance(element, list):
            raise TypeError(err)

    err = 'matrix must be a square matrix'
    for element in matrix:
        if len(element) != my_len:
            raise ValueError(err)

    if my_len == 1:
        return matrix[0][0]

    # base case
    if my_len == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return (a * d) - (b * c)

    # recursion
    det = 0
    for i, k in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        new_m = [[row[n] for n in range(my_len) if n != i] for row in rows]
        det += k * (-1) ** i * determinant(new_m)

    return det


def minor(matrix):
    """
    calculates the minor of a matrix:
    :param matrix: list of lists whose determinant should be calculated
    :return: minor of matrix
    """
    err = 'matrix must be a list of lists'
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError(err)

    for element in matrix:
        if not isinstance(element, list):
            raise TypeError(err)

    err = 'matrix must be a non-empty square matrix'
    my_len = len(matrix)
    if my_len == 1 and len(matrix[0]) == 0:
        raise ValueError(err)

    for element in matrix:
        if len(element) != my_len:
            raise ValueError(err)

    if my_len == 1:
        return [[1]]

    minor = []
    for i in range(my_len):
        minor.append([])
        for j in range(my_len):
            rows = [matrix[m] for m in range(my_len) if m != i]
            new_m = [[row[n] for n in range(my_len) if n != j] for row in rows]
            my_det = determinant(new_m)
            minor[i].append(my_det)

    return minor
