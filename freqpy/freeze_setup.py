import sys
from cx_Freeze import setup, Executable
import matplotlib
import scipy
import os

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["os", "wx", "matplotlib", "scipy", 
                                  "sklearn.utils.lgamma", 
                                  "sklearn.neighbors.typedefs",
                                  "sklearn.utils.sparsetools._graph_validation",
                                  "sklearn.utils.weight_vector"],
                     "include_files": ["SETTINGS", "Data", "Readme"],
                     "excludes": ["tkinter", 
                                  "collections.sys",
                                  "collections._weakref",
                                  "tcl"]}

bdist_msi_options = {
                    'upgrade_code': '{111111-1010101-666A}',
                    'add_to_path': True
                    }

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(  name = "FreqPy",
        version = "0.00001",
        description = "Analyze Data for Population Coding",
        options = {
                    "build_exe": build_exe_options,
                    "bdist_msi": bdist_msi_options
                    },
        executables = [Executable("freqgui.py", base=base)])