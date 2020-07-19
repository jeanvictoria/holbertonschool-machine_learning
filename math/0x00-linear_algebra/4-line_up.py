#!/usr/bin/env python3
"""4-line_up.py contain add_arrays function"""


def add_arrays(array1, array2):
    """adds two arrays element-wise"""
    if len(array1) != len(array2):
        return None
    return [array1[i] + array2[i] for i in range(0, len(array1), 1)]
