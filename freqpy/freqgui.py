#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pprint
import random
import sys

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

from sklearn.decomposition import PCA

class ImportFiles(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.create_buttons()
        self.create_listctrl()
        self.create_radios()
        self.__do_layout()
        self.neurons = list()
        self.conditions = list()

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Name")
        self.listCtrl.InsertColumn(1, "File Data Type")
        self.listCtrl.SetColumnWidth(0, 100)
        self.listCtrl.SetColumnWidth(1, 100)        

    def create_radios(self):
        self.RadioTitle = wx.StaticText(self, -1, label="Select Data Type:", style=wx.RB_GROUP)
        self.rb1 = wx.RadioButton(self, -1, 'Neural', (10, 10), style=wx.RB_GROUP)
        self.rb2 = wx.RadioButton(self, -1, 'Condition', (10, 30))
        self.Bind(wx.EVT_RADIOBUTTON, self.SetVal, id=self.rb1.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.SetVal, id=self.rb2.GetId())
        self.state = 'Neural'

    def SetVal(self, event):
        state1 = str(self.rb1.GetValue())
        state2 = str(self.rb2.GetValue())
        if state1 == 'True':
            self.state = 'Neural'
        elif state2 == 'True':
            self.state = 'Condition'

    def create_buttons(self):
        self.DataButton = wx.Button(self, -1, "Import File...")
        self.Bind(wx.EVT_BUTTON, self.on_add_file, self.DataButton)

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.listCtrl, 1, wx.EXPAND, 0)
        sizer_1.Add(self.RadioTitle, wx.ALIGN_LEFT)
        sizer_1.Add(self.rb1, wx.ALIGN_LEFT)
        sizer_1.Add(self.rb2, wx.ALIGN_LEFT)
        sizer_1.Add(self.DataButton, 0, wx.ALIGN_LEFT)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()

    def on_add_file(self, event):
        file_choices = "TXT (*.txt)|*.txt"
        dialog = wx.FileDialog(
        self, 
        message="Import file...",
        defaultDir=os.getcwd(),
        defaultFile="",
        wildcard=file_choices,
        style=wx.OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.listCtrl.Append([dialog.GetPath().split('\\')[-1], self.import_files.state])
            if self.state == 'Neural':
                self.neurons.append(dialog.GetPath())
            elif self.state == 'Condition':
                self.append(dialog.GetPath())

class LabelData(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.dpi = 200
        self.fig = Figure((5.0, 3.0), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.data_arr = dict()
        self.t = 0
        self.create_listctrl()
        self.create_neur_box()
        self.__do_layout()

    def load_data(self, filenames):
        for filename in filenames:
            data = np.loadtxt(filename)
            if data.shape[0] > data.shape[1]:
                data = data.T
            for neurno in range(data.shape[0]):
                self.neur_box.InsertItems(pos=neurno, items=[str(neurno)])
            self.data_arr.update({filename: data})
        if self.t == 0:
            self.init_plot()
            self.t += 1

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Path")
        self.listCtrl.InsertColumn(1, "File Data Type")
        self.listCtrl.SetColumnWidth(0, 200)
        self.listCtrl.SetColumnWidth(1, 100)

    def create_neur_box(self):
        self.neur_title = wx.StaticText(self, -1, "Select Neuron:")
        self.neur_box = wx.ListBox(self, -1, 
                                   style=wx.LB_SINGLE|wx.LB_NEEDED_SB, size=(100,500))
        self.neur_box.Bind(wx.EVT_LISTBOX, self.plot_selected)

    def plot_selected(self, event):
        sel_neuron = self.neur_box.GetSelection()
        self.plot_data.set_xdata(np.arange(self.data_arr[self.data_arr.keys()[0]].shape[1]))
        self.plot_data.set_ydata(self.data_arr[self.data_arr.keys()[0]][sel_neuron, :])
        self.canvas.draw()

    def init_plot(self):
        self.axes = self.fig.add_subplot(111)
        self.axes.set_axis_bgcolor('white')
        self.axes.set_title('Frequency Response', size=10)
        self.axes.set_xlabel('time (sec)')
        self.axes.set_ylabel('frequency (hz)')
        
        plt.setp(self.axes.get_xticklabels(), fontsize=5)
        plt.setp(self.axes.get_yticklabels(), fontsize=5)

        self.plot_data = self.axes.plot(
            self.data_arr[self.data_arr.keys()[0]][0, :],
            linewidth=0.3,
            color=(0, 0, 0),
            )[0]
        self.canvas.draw()

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_1.Add(self.listCtrl, 0, wx.ALIGN_LEFT, 5)
        sizer_2 = wx.BoxSizer()
        sizer_2.Add(self.canvas, 2, wx.ALIGN_CENTER)
        sizer_3 = wx.BoxSizer(wx.VERTICAL)
        sizer_3.Add(self.neur_title)
        sizer_3.Add(self.neur_box)
        sizer_2.Add(sizer_3)
        sizer_1.Add(sizer_2)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()

class Analyze(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.t = 0
        self.dpi = 200
        self.fig = Figure((5.0, 3.0), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.data_arr = dict()
        self.neurons = list()
        self.conditions = list()

    def load_data(self, filenames):
        for filename in filenames:
            data = np.loadtxt(filename)
            if data.shape[0] > data.shape[1]:
                data = data.T
            pca = PCA(n_components=3)
            pca.fit(data.T)
            data = pca.transform(data.T)
            self.data_arr.update({filename: data})
        if self.t == 0:
            self.init_plot()
            self.t += 1

    def init_plot(self):
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_axis_bgcolor('white')
        self.axes.set_title('Cluster Analysis', size=10)
        self.axes.set_xlabel('PC1',size=5)
        self.axes.set_ylabel('PC2',size=5)
        self.axes.set_zlabel('PC3',size=5)
        
        plt.setp(self.axes.get_xticklabels(), fontsize=4)
        plt.setp(self.axes.get_yticklabels(), fontsize=4)
        plt.setp(self.axes.get_zticklabels(), fontsize=4)

        init_dat = self.data_arr[self.data_arr.keys()[0]]
        init_labels = np.loadtxt('pdat_labels.txt')
        init_labels = [(1/lab, lab*0.12, lab*0.1, 1.0) for lab in init_labels]
        self.axes.scatter(init_dat[:, 0], init_dat[:, 1], init_dat[:, 2],
                          c=init_labels)
        self.canvas.draw()

    def ellipsoid_creation(self, labels, data):


class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="FreqPy", size=(1500, 800))

        self.neurons = list()
        self.conditions = list()

        p = wx.Panel(self)
        nb = wx.Notebook(p)

        self.import_files = ImportFiles(nb)
        self.import_files.DataButton.Bind(wx.EVT_BUTTON, self.on_add_file)
        self.label_data = LabelData(nb)
        self.analyze = Analyze(nb)

        nb.AddPage(self.import_files, "Import Files")
        nb.AddPage(self.label_data, "Label Data")
        nb.AddPage(self.analyze, "Analyze")

        sizer = wx.BoxSizer()
        sizer.Add(nb, 1, wx.EXPAND)
        p.SetSizer(sizer)
        self.Layout()

    def on_add_file(self, event):
        file_choices = "TXT (*.txt)|*.txt"
        dialog = wx.FileDialog(
        self, 
        message="Import file...",
        defaultDir=os.getcwd(),
        defaultFile="",
        wildcard=file_choices,
        style=wx.OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.import_files.listCtrl.Append([dialog.GetPath().split('\\')[-1], 
                                               self.import_files.state])
            self.label_data.listCtrl.Append([dialog.GetPath(), 
                                             self.import_files.state])
            if self.import_files.state == 'Neural':
                self.import_files.neurons.append(dialog.GetPath())
                self.label_data.load_data(self.import_files.neurons)
                self.analyze.load_data(self.import_files.neurons)
            elif self.import_files.state == 'Condition':
                self.import_files.append(dialog.GetPath())

        dialog.Destroy()

if __name__ == '__main__':
    app = wx.App()
    app.frame = MainFrame()
    app.frame.Show()
    app.MainLoop()
