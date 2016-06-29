#!/usr/bin/python
# -*- coding: utf-8 -*-
import shutil
import os
import pprint
import random
import sys
import itertools
import time
import shlex
from operator import itemgetter

import signal
from multiprocessing import Process
from threading import Thread
import subprocess

import json
import pickle

import wx
from wx.lib.pubsub import setuparg1
from wx.lib.pubsub import pub as Publisher
# from wx.lib.pubsub import pubsubconf
import wx.lib.newevent

import numpy as np
# from scipy.stats import entropy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, cdist
# from scipy.signal import spline_filter, gauss_spline
# from scipy.ndimage.filters import gaussian_filter
# from scipy.ndimage.interpolation import spline_filter

import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
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

from sklearn import cluster
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GMM
from sklearn.svm import OneClassSVM

from smoothing import bezier, exponential
from utils import *
# from generic_menu import GenericMenu