from extimports import *

class ReadMe(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.title_ = wx.StaticText(self, -1, 
                                    "ReadMe Doc", size=(75, 1))
        self.t1 = wx.TextCtrl(self, -1, size=(800, 500), style=wx.TE_MULTILINE|wx.TE_READONLY|wx.TE_AUTO_URL)
        self.populate_readme()
        self.t1.SetInsertionPoint(0)

        self.__do_layout()

    def populate_readme(self):
        with open('README.md', 'r') as rmf:
            rlines = rmf.readlines()
        for rl in rlines:
            self.t1.AppendText(rl)

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)

        # sizer_1.Add(self.title_, 0, 0, 0)
        
        sizer_1.Add(self.t1, 1, wx.EXPAND|wx.ALIGN_CENTER|wx.LEFT|wx.RIGHT, 0)

        sizer_1.SetSizeHints(self)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()