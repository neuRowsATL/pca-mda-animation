from extimports import *

class ImportFiles(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.create_buttons()
        self.create_listctrl()
        # self.create_slider()
        self.__do_layout()

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Name")
        self.listCtrl.InsertColumn(1, "File Data Type")
        self.listCtrl.SetColumnWidth(0, 100)
        self.listCtrl.SetColumnWidth(1, 100)

    def create_buttons(self):
        self.DataButton = wx.Button(self, -1, "Select Data Folder")
        self.SaveButton = wx.Button(self, -1, "Select Export Folder")
        # self.ResButton = wx.Button(self, -1, "Select Resolution")

    def create_slider(self):
        self.titler = wx.StaticText(self, -1, 
                                    "Choose data resolution.\n Data must be >= 1000 data points:", 
                                    (25, 10))
        self.t1 = wx.TextCtrl(self, -1, "1000")

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.listCtrl, 1, wx.EXPAND, 0)
        hsize = wx.BoxSizer(wx.HORIZONTAL)
        hsize2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_1.Add(self.titler, 0, wx.ALIGN_LEFT)
        # hsize2.Add(self.t1, 0, wx.ALIGN_LEFT)
        # hsize2.Add(self.ResButton, 0, wx.ALIGN_RIGHT)
        hsize.Add(self.DataButton, 0, wx.ALIGN_LEFT|wx.ALL)
        hsize.Add(self.SaveButton, 0, wx.ALIGN_RIGHT|wx.ALL)
        sizer_1.Add(hsize2)
        sizer_1.Add(hsize)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()