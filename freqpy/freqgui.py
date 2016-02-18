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
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar

import numpy as np
import pylab

class ImportFiles(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.create_buttons()
        self.create_listctrl()
        self.create_radios()
        self.__do_layout()

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Names")
        self.listCtrl.InsertColumn(1, "File Type")
        self.listCtrl.SetColumnWidth(0, 100)
        self.listCtrl.SetColumnWidth(1, 100)        

    def create_radios(self):
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
            self.listCtrl.Append([dialog.GetPath().split('\\')[-1], self.state])
        dialog.Destroy()

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.listCtrl, 1, wx.EXPAND, 0)
        sizer_1.Add(self.rb1, wx.ALIGN_LEFT)
        sizer_1.Add(self.rb2, wx.ALIGN_LEFT)
        sizer_1.Add(self.DataButton, 0, wx.ALIGN_CENTER)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()

class PageTwo(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        t = wx.StaticText(self, -1, "This is a PageTwo object", (40,40))

class PageThree(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        t = wx.StaticText(self, -1, "This is a PageThree object", (60,60))

class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="FreqPy")

        # Here we create a panel and a notebook on the panel
        p = wx.Panel(self)
        nb = wx.Notebook(p)

        # create the page windows as children of the notebook
        import_files = ImportFiles(nb)
        page2 = PageTwo(nb)
        page3 = PageThree(nb)

        # add the pages to the notebook with the label to show on the tab
        nb.AddPage(import_files, "Import Files")
        nb.AddPage(page2, "Label Data")
        nb.AddPage(page3, "Analyze")

        # finally, put the notebook in a sizer for the panel to manage
        # the layout
        sizer = wx.BoxSizer()
        sizer.Add(nb, 1, wx.EXPAND)
        p.SetSizer(sizer)

if __name__ == '__main__':
    app = wx.App()
    app.frame = MainFrame()
    app.frame.Show()
    app.MainLoop()
