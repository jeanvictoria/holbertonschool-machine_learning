#!/usr/bin/env python3
"""3-flip_me_over.py contain matrix_transpose function"""


def matrix_transpose(m):
    """returns the transpose of a 2D matrix, matrix"""
    if not isinstance(m, list) or not len(m) > 0 or not isinstance(m[0], list):
        return None
    transpose = []
    rows, cols = len(m), len(m[0])
    for c in range(cols):
        transpose.append([m[r][c] for r in range(rows)])
    return transpose
