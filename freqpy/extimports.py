#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pprint
import random
import sys
import itertools
import time
from operator import itemgetter

from multiprocessing import Pool, cpu_count
import subprocess


import wx

from scipy.stats import entropy


import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar

import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib import cm as CM
import matplotlib.ticker as plticker 

import numpy as np
import pylab as plt
from ellipsoid import EllipsoidTool
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GMM
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM


from smoothing import bezier