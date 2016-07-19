from extimports import *

class ImportFiles(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)


        
        self.create_titles()
        self.create_buttons()
        self.create_listctrl()
        self.__do_layout()

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Name")
        self.listCtrl.InsertColumn(1, "File Data Type")
        self.listCtrl.SetColumnWidth(0, 100)
        self.listCtrl.SetColumnWidth(1, 100)

    def create_titles(self):
        self.FoldTitle = wx.StaticText(self, -1, "Choose Folders")
        self.ResTitle = wx.StaticText(self, -1, "Choose Resolution")
        self.ImportTitle = wx.StaticText(self, -1, "Import Data")

    def create_buttons(self):
        self.DataButton = wx.Button(self, -1, "Select Data Folder")
        self.SaveButton = wx.Button(self, -1, "Select Export Folder")

        self.ResButton = wx.Button(self, -1, "Set Resolution")
        
        self.DataImpButton = wx.Button(self, -1, "Import Spikes")
        self.LabelsImpButton = wx.Button(self, -1, "Import Labels")
        self.WaveformImpButton = wx.Button(self, -1, "Import Waveform")

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        hsize = wx.BoxSizer(wx.HORIZONTAL)
        hsize.Add(self.listCtrl, 2, wx.EXPAND, 0)
        sizer_1.Add(hsize, 0, wx.EXPAND|wx.RIGHT)

        sizer_1.Add(wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL, size=(800,1)))
        sizer_1.AddSpacer(10)

        sizer_1.Add(self.FoldTitle, 0, wx.ALIGN_CENTER)
        hsize2 = wx.BoxSizer(wx.HORIZONTAL)
        hsize2.Add(self.DataButton, 1, wx.ALIGN_CENTER, 0)
        hsize2.Add(self.SaveButton, 1, wx.ALIGN_CENTER, 0)
        sizer_1.Add(hsize2, 0, wx.ALIGN_CENTER)
        
        sizer_1.AddSpacer(10)
        sizer_1.Add(wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL, size=(800,1)))
        sizer_1.AddSpacer(10)

        sizer_1.Add(self.ResTitle, 0, wx.ALIGN_CENTER)
        hsize_2 = wx.BoxSizer(wx.HORIZONTAL)
        hsize_2.Add(self.ResButton, 0, wx.ALIGN_CENTER)
        sizer_1.Add(hsize_2, 0, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(10)
        sizer_1.Add(wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL, size=(800,1)))
        sizer_1.AddSpacer(10)

        sizer_1.Add(self.ImportTitle, 0, wx.ALIGN_CENTER)
        hsize3 = wx.BoxSizer(wx.HORIZONTAL)
        hsize3.Add(self.DataImpButton, 0, wx.ALIGN_CENTER)
        hsize3.Add(self.LabelsImpButton, 0, wx.ALIGN_CENTER)
        hsize3.Add(self.WaveformImpButton, 0, wx.ALIGN_CENTER)
        sizer_1.Add(hsize3, 0, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(10)
        sizer_1.Add(wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL, size=(800,1)))
        sizer_1.AddSpacer(10)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()