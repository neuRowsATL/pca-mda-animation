#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pprint
import random
import sys
import itertools

import wx

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar

import numpy as np
import pylab as plt
from ellipsoid import EllipsoidTool
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize