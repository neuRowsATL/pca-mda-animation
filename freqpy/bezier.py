import numpy as np
from scipy.special import binom

""" Citation: https://gist.github.com/Juanlu001/7284462
"""

def bernstein(n, k):
    coeff = binom(n, k)
    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)
    return _bpoly

def bezier(points, res=1000, dim=3):
    N = len(points)
    t = np.linspace(0, 1, num=res)
    curve = np.zeros((res, dim))
    for ii in range(N):
        curve += np.outer(bernstein(N - 1, ii)(t), points[ii])
    return curve

# points = np.random.normal(loc=0.0, scale=1.1, size=(1000, 3))
# curve = bezier(points)