# DataView Traces to Waveform Vis.

import numpy as np
import matplotlib.pyplot as plt
import os


def waveform_convert(filename, nr_trace, toplot=False, tosave=True, outname='waveform.txt'):
    def iter_loadtxt(filename, delimiter='\t', skiprows=6, dtype=np.float32):
        def iter_func():
            with open(filename, 'r') as infile:
                for _ in range(skiprows):
                    next(infile)
                for line in infile:
                    line = line.rstrip().split(delimiter)
                    for item in line:
                        yield dtype(item)
            iter_loadtxt.rowlength = len(line)
        data = np.fromiter(iter_func(), dtype=dtype)
        data = data.reshape((-1, iter_loadtxt.rowlength))
        return data
    wv = iter_loadtxt(filename, delimiter='\t', skiprows=6)
    if toplot is True:
        plt.plot(wv[:, nr_trace])
        plt.show()
    if tosave is True:
        np.savetxt(outname, wv[:, nr_trace])
    return wv[:, nr_trace]

def load_waveform(filename, trace):
    def iter_loadtxt(filename, delimiter='\t', skiprows=6, dtype=float):
        def iter_func():
            with open(filename, 'r') as infile:
                for _ in range(skiprows):
                    next(infile)
                for line in infile:
                    line = line.rstrip().split(delimiter)
                    for item in line:
                        yield dtype(item)
            iter_loadtxt.rowlength = len(line)
        data = np.fromiter(iter_func(), dtype=dtype)
        data = data.reshape((-1, iter_loadtxt.rowlength))
        return data
    return iter_loadtxt(filename)[:, trace]

def iter_loadtxt(filename, delimiter='\t', skiprows=6, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)
    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

def gen_from_iter(filename, delimiter='\t', skiprows=0, dtype=float):
    with open(filename, 'r') as infile:
        for _ in range(skiprows):
            next(infile)
        for line in infile:
            line = line.rstrip().split(delimiter)
            for item in line:
                # print(dtype(item))
                yield dtype(item)

def waveform_compress(filename, trace, n=1000):
    reduced = list()
    wv = load_waveform(filename, trace)
    s = int(len(wv)/n)
    for i in range(int(n)):
        slice_ = wv[i*s:(i+1)*s]
        reduced.append(np.mean(slice_))
    return np.array(reduced, dtype=np.float32)


