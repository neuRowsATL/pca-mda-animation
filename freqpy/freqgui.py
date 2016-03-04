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
from ellipsoid import EllipsoidTool
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

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
        self.cond_arr = dict()
        self.t = 0
        self.create_listctrl()
        self.create_neur_box()
        self.__do_layout()

    def load_data(self, filenames):
        for filename in filenames:
            data = np.loadtxt(filename)
            if data.shape[0] > data.shape[1]:
                data = data.T
            self.data_arr.update({filename: data})
            self.neur = filename
        if self.t == 0:
            self.init_plot()
            self.t += 1

    def load_conditions(self, filenames):
        for filename in filenames:
            conds = np.loadtxt(filename)
            self.cond_arr.update({filename: conds})

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Path")
        self.listCtrl.InsertColumn(1, "File Data Type")
        self.listCtrl.SetColumnWidth(0, 200)
        self.listCtrl.SetColumnWidth(1, 100)
        self.listCtrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.populate_neur_box)

    def create_neur_box(self):
        self.neur_title = wx.StaticText(self, -1, "Select Neuron:")
        self.neur_box = wx.ListBox(self, -1, 
                                   style=wx.LB_SINGLE|wx.LB_NEEDED_SB, size=(100,100))
        self.neur_box.Bind(wx.EVT_LISTBOX, self.plot_selected)

    def populate_neur_box(self, event):
        neur, cond = self.separate_selected()
        if self.neur != neur:
            self.neur = neur
            for idx in range(self.neur_box.GetCount()):
                self.neur_box.Delete(idx)
            neurfile = neur[0]
            for neurno in range(self.data_arr[neurfile].shape[0]):
                self.neur_box.InsertItems(pos=neurno, items=[str(neurno)])

    def plot_selected(self, event):
        neur, cond = self.separate_selected()
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        if len(neur) > 0:
            sel_neuron = self.neur_box.GetSelection()
            self.plot_data.set_xdata(np.arange(self.data_arr[neur[0]].shape[1]))
            self.plot_data.set_ydata(self.data_arr[neur[0]][sel_neuron, :])
            self.axes.plot()
            if len(cond) > 0:
                current_cond = self.cond_arr[cond[0]]
                current_neuron_range = np.arange(self.data_arr[neur[0]].shape[1])
                classes = np.arange(min(current_cond), max(current_cond)+1)
                for class_ in classes:
                    selected_class = current_neuron_range[current_cond==class_]
                    self.axes.axvspan(min(selected_class), max(selected_class), facecolor=color_list[int(class_)-1], 
                        alpha=0.5)
            self.canvas.draw()
        else:
            print('You must select a Neural file to plot frequency.')

    def separate_selected(self):
        neur = list()
        cond = list()
        if self.listCtrl.GetSelectedItemCount() != 0:
            sel_files = self.get_selected()
            for sel in sel_files:
                filename = self.listCtrl.GetItemText(sel, 0)
                filetype = self.listCtrl.GetItemText(sel, 1)
                if filetype == 'Condition':
                    cond.append(filename)
                elif filetype == 'Neural':
                    neur.append(filename)
        return neur, cond

    def get_selected(self):
        current = -1
        selection = list()
        while True:
            next_ = self.listCtrl.GetNextSelected(current)
            if next_ == -1:
                return selection
            selection.append(next_)
            current = next_

    def init_plot(self):
        self.axes = self.fig.add_subplot(111)
        self.axes.set_axis_bgcolor('white')
        self.axes.set_title('Frequency Response', size=10)
        self.axes.set_xlabel('time (sec)', size=5)
        self.axes.set_ylabel('frequency (hz)', size=5)
        
        plt.setp(self.axes.get_xticklabels(), fontsize=5)
        plt.setp(self.axes.get_yticklabels(), fontsize=5)

        self.plot_data = self.axes.plot(
            self.data_arr[self.data_arr.keys()[0]][0, :],
            linewidth=0.3,
            color=(0, 0, 0),
            )[0]
        self.canvas.draw()

    def __do_layout(self):
        sizer_1 = wx.BoxSizer()
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        sizer_2.Add(self.listCtrl, 0, wx.ALIGN_LEFT, 5)
        sizer_2.Add(self.neur_title, 0, wx.ALIGN_LEFT, 1)
        sizer_2.Add(self.neur_box, 0, wx.ALIGN_LEFT)
        sizer_1.Add(sizer_2)
        sizer_1.Add(self.canvas)
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
        self.cond_arr = dict()
        self.neurons = list()
        self.conditions = list()
        self.__do_layout()

    def load_data(self, filenames):
        for filename in filenames:
            data = np.loadtxt(filename)
            if data.shape[0] > data.shape[1]:
                data = data.T
            pca = PCA(n_components=3)
            pca.fit(data.T)
            data = normalize(pca.transform(data.T))
            self.data_arr.update({filename: data})
        if self.t == 0:
            self.init_plot()
            self.t += 1

    def load_conditions(self, filenames):
        for filename in filenames:
            conds = np.loadtxt(filename)
            self.cond_arr.update({filename: conds})

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
        try:
            init_labels = self.cond_arr[self.cond_arr.keys()[0]]
        except IndexError:
            init_labels = np.loadtxt('pdat_labels.txt')
        labelled_data = self.class_creation(init_labels, init_dat)
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        for class_label in labelled_data.keys():

            current_class = labelled_data[class_label]

            pca = PCA(n_components=3)
            pca.fit(current_class)
            projected_class = normalize(pca.transform(current_class))
            projected_class = current_class*projected_class

            x = projected_class[:, 0]
            y = projected_class[:, 1]
            z = projected_class[:, 2]

            self.axes.scatter(x, y, z, c=color_list[int(class_label)-1], marker='.', edgecolor='k')
            center, radii, rotation = EllipsoidTool().getMinVolEllipse(projected_class)
            """ 
            EllipsoidTool from:
            https://github.com/minillinim/ellipsoid 
            """
            EllipsoidTool().plotEllipsoid(center, radii, rotation, ax=self.axes, plotAxes=True, 
                                        cageColor=color_list[int(class_label)-1], cageAlpha=0.5)
        self.canvas.draw()

    def pca_selected(self, event):
        pass

    def mda_selected(self, event):
        pass

    def kmeans_selected(self, event):
        pass

    def class_creation(self, labels, data):
        classes = dict()
        for label in range(int(min(labels)), int(max(labels))+1):
            classes[label] = data[labels==label,:]
        return classes

    def __do_layout(self):
        sizer_1 = wx.BoxSizer()
        sizer_1.Add(self.canvas)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()

class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="FreqPy", size=(900, 900))

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
                self.import_files.conditions.append(dialog.GetPath())
                self.analyze.load_conditions(self.import_files.conditions)
                self.label_data.load_conditions(self.import_files.conditions)

        dialog.Destroy()

if __name__ == '__main__':
    app = wx.App()
    app.frame = MainFrame()
    app.frame.Show()
    app.MainLoop()
