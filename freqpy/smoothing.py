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

def exponential(x, alpha=0.3):
    N = x.shape[0]
    S_t = np.zeros_like(x)
    S_t[0, :] = x[0, :]
    for ii in range(1, N):
        S_t[ii, :] = alpha*x[ii-1, :] + (1.0-alpha)*S_t[ii-1, :]
    return S_t
