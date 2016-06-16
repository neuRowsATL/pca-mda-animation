from extimports import *

class FormatFileNames(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.create_buttons()
        self.create_listctrl()
        self.create_text()
        self.create_input()
        self.__do_layout()

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Name")
        self.listCtrl.InsertColumn(1, "File Data Type")
        self.listCtrl.SetColumnWidth(0, 100)
        self.listCtrl.SetColumnWidth(1, 100)

    def create_buttons(self):
        self.DataButton = wx.Button(self, -1, "Select Spike On/Off Files")
        self.LabelsButton1 = wx.Button(self, -1, "Select Labels File (integer labels)")
        self.LabelsButton2 = wx.Button(self, -1, "Select Labels File (event times)")
        self.ResButton = wx.Button(self, -1, "Select Resolution")

    def create_text(self):
        self.titler = wx.StaticText(self, -1, 
                                    "Use this tab to help format your data folder before importing."+\
                                    "\nFollow the readme for more info.", 
                                    (50, 10))

    def create_input(self):
        self.t1 = wx.TextCtrl(self, -1, "1000")

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.titler, 0, wx.ALIGN_CENTER)
        sizer_1.Add(self.listCtrl, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()