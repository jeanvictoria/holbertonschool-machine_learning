#!/usr/bin/env python3
"""Returns derivative polynomial array"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if type(poly) is not list:
        return None
    elif len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        derivate = [0] * (len(poly) - 1)
        for i in range(len(poly) - 1):
            if (type(poly[i]) is not int):
                return None
            derivate[i] = poly[i + 1] * (i + 1)
    return derivate
