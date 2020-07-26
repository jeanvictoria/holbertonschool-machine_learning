#!/usr/bin/env python3
"""contain the poly_integral(poly, C=0) function"""


def poly_integral(poly, C=0):
    """calculate the integral of a polynomial"""
    if not type(poly) is list or len(poly) == 0 or\
            not (type(poly[0]) is int or type(poly[0] is float)):
        return None
    if not (type(C) is int or type(C) is float):
        return None

    integral = [C]
    for i, coef in enumerate(poly, 1):
        if coef % i == 0:
            integral.append(coef // i)
        else:
            integral.append(coef / i)
    for i in range(len(integral) - 1, 0, -1):
        if integral[i] == 0:
            integral.pop(i)
        else:
            break
    return integral
