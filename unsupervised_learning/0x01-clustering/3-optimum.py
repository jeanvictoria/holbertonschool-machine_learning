
#!/usr/bin/env python3
"""Function calculates optimum num clusters"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
     calculate optimun num clusters
    :param X: np.ndarray, (n, d)
    :param kmin: pos int, contain min num clusters
    :param kmax: pos int contain max num clusters
    :param iterations: pos int contains max num inters
    :return: results, d_vars, or None, None
    """
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if not isinstance(kmax, int) or kmax < 1:
        return None, None
    if kmin >= kmax:
        return None, None
    try:
        results = []
        d_vars = []
        kmin_km = kmeans(X, kmin)
        kmin_var = variance(X, kmin_km)
        for m in range(kmin, kmax + 1):
            C, clss = kmeans(X, k)
            results.append((C, clss))
            d_vars.append(kmin_var - variance(X, C))
        return (results, d_vars)
    except Exception:
        return None, None
