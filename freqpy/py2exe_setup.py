# setup.py
from distutils.core import setup
import py2exe

import glob

import numpy
import sklearn
import matplotlib
import mpl_toolkits
from wx.lib.pubsub import setuparg1


packages = ['wx.lib.pubsub']
includes = ['matplotlib.backends.backend_wxagg', 'matplotlib.figure', 'numpy', 
            'sklearn', 'sklearn.utils.*', 'sklearn.neighbors.typedefs', 'sklearn.utils.sparsetools._graph_validation',
            
            'pylab', 'mpl_toolkits', 'scipy.sparse.csgraph._validation', 'scipy',
            'scipy.integrate', 'scipy.special.*', 'scipy.linalg.*']

excludes = ['_gtkagg', '_tkagg']

dll_excludes = ['libgdk-win32-2.0-0.dll', 'libgobject-2.0-0.dll', 'tcl84.dll',
                'tk84.dll']

data_files = matplotlib.get_py2exe_datafiles()

# setup(windows=['freqgui.py'])

setup(
    options = {"py2exe": {"compressed": 0, 
                          "optimize": 0,
                          "includes": includes,
                          "excludes": excludes,
                          "packages": packages,
                          "dll_excludes": dll_excludes,
                          "bundle_files": 3,
                          "dist_dir": "dist",
                          "xref": False,
                          "skip_archive": False,
                          "ascii": False,
                          "custom_boot_script": '',
                         }
              },
    windows=['freqgui.py'],
    data_files = data_files
)