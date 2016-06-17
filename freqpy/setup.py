"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup
import numpy
import sklearn
import matplotlib
import mpl_toolkits
import wx
import FreqPy
import freqgui
import macholib_patch

APP = ['FreqPy.py']
DATA_FILES = ['Data']
includes = ['matplotlib.backends.backend_wxagg', 'matplotlib.figure', 'numpy', 
            'sklearn', 'sklearn.utils.*', 'sklearn.neighbors.typedefs', 
            'sklearn.utils.sparsetools._graph_validation',
            'pylab', 'mpl_toolkits', 'scipy.sparse.csgraph._validation', 'scipy',
            'scipy.integrate', 'scipy.special.*', 'scipy.linalg.*',
            'wx.*']
OPTIONS = {'argv_emulation': True, 'includes': includes}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
