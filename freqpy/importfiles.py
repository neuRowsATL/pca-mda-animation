from extimports import *

class ImportFiles(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.create_buttons()
        self.create_listctrl()
        self.__do_layout()
        self.neurons = list()
        self.conditions = list()

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Name")
        self.listCtrl.InsertColumn(1, "File Data Type")
        self.listCtrl.SetColumnWidth(0, 100)
        self.listCtrl.SetColumnWidth(1, 100)

    def create_buttons(self):
        self.DataButton = wx.Button(self, -1, "Import Data Folder")

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.listCtrl, 1, wx.EXPAND, 0)
        sizer_1.Add(self.DataButton, 0, wx.ALIGN_LEFT)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()