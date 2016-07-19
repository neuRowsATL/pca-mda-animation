import numpy as np
from scipy.special import binom
import time
from itertools import combinations_with_replacement as cwr, starmap
import matplotlib.pyplot as plt
from decimal import Decimal

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

def casteljau(points, res, dim, deg=3):
    N = Decimal(str(len(points)))
    t = np.linspace(0, 1, num=res)
    t = [Decimal(t) for t in t.tolist()]
    if dim > 1: points = [[Decimal(str(p2)) for p2 in p1] for p1 in points.tolist()]
    else: points = [Decimal(str(p1)) for p1 in points.tolist()]
    curve = np.zeros((res, dim))
    curve = [[Decimal(str(c2)) for c2 in c1] for c1 in curve.tolist()]
    for ii in range(len(points)):
        coeff = Decimal(str(binom(len(points)-1, ii)))
        bp = [coeff * (ti ** Decimal(str(ii))) * (Decimal('1')-ti) ** ((N-Decimal('1'))-ii) for ti in t]
        pi = points[ii]
        for mm in range(len(bp)):
            for nn in range(len(pi)):
                curve[mm][nn] += [cr[0]*cr[1] for cr in zip(bp, pi)]
    return curve

class ExpSmooth:
    def __init__(self, x, **kwargs):
        self.x = x
        self.options = {
                    'gamma': 0.1,
                    'alpha': 0.1,
                    'beta': 0.1,
                    'res': x.shape[0],
                    'verbose': False
                }
        self.options.update(kwargs)

    def __modulo_value(self, key):
        return self.options[key] % 0.999999999999

    def exponential_single(self, alpha=None):
        x = self.x.copy()
        if alpha is None:
            alpha = self.__modulo_value('alpha')
        N = x.shape[0]
        S_t = np.zeros_like(x)
        S_t[0, :] = x[0, :]
        for ii in range(1, N):
            S_t[ii, :] = alpha*x[ii-1, :] + (1.0-alpha)*S_t[ii-1, :]
        return alpha, S_t, np.linalg.norm(x - S_t)

    def exponential_double(self, *args, **kwargs):
        x = self.x.copy()
        N = x.shape[0]
        if kwargs is not None:
            if 'verbose' in kwargs.keys(): self.options['verbose'] = kwargs['verbose']
        if len(args) > 1:
            alpha, gamma = args
        elif len(args) == 1:
            param = args[0]
            alpha, gamma = param
        if alpha is None:
            alpha = self.__modulo_value('alpha')
        if gamma is None:
            gamma = self.__modulo_value('gamma')
        b1 = np.mean(np.diff(x, 0), 0)
        S_t = np.zeros_like(x)
        b_t = np.zeros_like(x)
        S_t[0, :] = x[0, :]
        b_t[0, :] = x[1, :] - x[0, :] # b1
        for ii in range(1, N):
            S_t[ii, :] = alpha*x[ii, :] + (1.0-alpha)*(S_t[ii-1, :] + b_t[ii-1, :])
            b_t[ii, :] = gamma*(S_t[ii, :] - S_t[ii-1, :]) + (1.0 - gamma)*b_t[ii-1, :]
        MSE = (np.tanh(np.linalg.norm(np.subtract(x, S_t))) + 1.0) / 2.0
        if self.options['verbose'] == True:
            print(alpha, gamma, MSE)
        return S_t, alpha, gamma, MSE

    def exponential_triple(self, *args):
        pass
        if len(args) > 1:
            alpha, beta, gamma = args
        elif len(args) == 1:
            param = args[0]
            alpha, beta, gamma = param
        if alpha is None:
            alpha = self.__modulo_value('alpha')
        if gamma is None:
            gamma = self.__modulo_value('gamma')
        if beta is None:
            beta = self.__modulo_value('beta')
        x = self.x.copy()
        N = x.shape[0]
        L = 4 ### Figure out how to find # of periods
        S_t = np.zeros_like(x)
        S_t[0, :] = x[0, :]
        b_t = np.zeros_like(x)
        I_t = np.zeros_like(x)
        for ii in range(1, N):
            S_t[ii, :] = alpha*(x[ii, :] / I_t[ii-L, :]) + (1.00 - alpha)*(S_t[ii-1, :] + b_t[ii-1, :])
            b_t[ii, :] = gamma*(S_t[ii, :] - S_t[ii-1, :]) + (1.00 - gamma)*b_t[ii-1, :]
            I_t[ii, :] = beta*(x[ii, :] / S_t[ii, :]) + (1.00 - beta)*I_t[ii-L, :]

    def fit(self, order='1', search_size=10, bounds=(0.000, 0.999), verbose=False, method='starmap'):
        if verbose == True: self.options['verbose'] = True
        if verbose == False: self.options['verbose'] = False
        x = self.x.copy()
        pick_alg = {'1': self.exponential_single,
                    '2': self.exponential_double,
                    '3': self.exponential_triple}
        smoother = pick_alg[order]
        search_range = np.linspace(bounds[0], bounds[1], search_size)

        t0 = time.time()
        if method == 'starmap':
            comps = starmap(smoother, cwr(search_range, 2))
        # elif method == 'imap':
            # comps = imap(smoother, cwr(search_range, 2))
        optimal_values = min(comps, key=lambda getter: getter[-1])
        t1 = time.time()

        total_min = int(round(t1 - t0, 3) / 60)
        total_sec = round(t1 - t0, 3) - total_min*60.0
        print("best alpha: %.3f, best gamma: %.3f, mean squared error: %.3f"\
              % (optimal_values[1], optimal_values[2], optimal_values[-1]))
        print("Total time: %.3f minutes and %.3f seconds." % (total_min, total_sec))
        return optimal_values


def test_exp():
    # x = np.random.normal(0, 0.1, (1000, 2))
    x = np.vstack((np.logspace(0.01, 0.9, 200), np.linspace(0, 100, 200))).T
    es = ExpSmooth(x)
    opt = es.fit(order='2', search_size=10, bounds=(0.000, 0.999), method='imap')
    plt.figure()
    ax2 = plt.subplot()
    ax2.plot(x[:, 0], color='b')
    ax1 = plt.subplot()
    ax1.plot(opt[0][:, 0], color='y', alpha=0.5, lw=5.0)
    plt.show()

def test_doub():
    # x = np.random.normal(0, 0.1, (1000, 2))
    x = np.vstack((np.logspace(0.01, 0.9, 200), np.linspace(0, 100, 200))).T
    es = ExpSmooth(x)
    opt = es.exponential_double(0.3, 0.5, verbose=True)
    ax = plt.subplot()
    ax.plot(x[:, 0], color='b')
    ax1 = plt.subplot()
    ax1.plot(opt[0][:, 0], color='y', alpha=0.5, lw=5.0)
    plt.show()

if __name__ == '__main__':
    test_exp()
    # test_doub()