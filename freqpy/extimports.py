#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pprint
import random
import sys
import itertools
import time
from multiprocessing import Pool
from threading import Thread
import subprocess

import wx

from scipy.stats.mstats import zscore

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib import animation

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Text
from matplotlib.lines import Line2D

import numpy as np
import pylab as plt
from ellipsoid import EllipsoidTool
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans