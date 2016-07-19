from extimports import *

class MyProgressDialog(wx.Dialog):
    
    def __init__(self):
        """Constructor"""
        wx.Dialog.__init__(self, None, title="Exporting Video... ")
        self.count = 0

        self.rng = 0
 
        self.progress = wx.Gauge(self)

        self.text = wx.StaticText(self, wx.ID_ANY, label="Completed: ")
        self.text.Wrap(-1)
 
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(self.progress, 0, wx.EXPAND, border=1)
        hs = wx.BoxSizer(wx.HORIZONTAL)
        hs.Add(self.text, 0, wx.ALIGN_CENTER, border=0)
        sizer.Add(hs, 0, wx.ALIGN_CENTER|wx.EXPAND, border=1)
        self.SetSizer(sizer)

        # pubsubconf.transitionV1ToV3('msg', step=2)
        Publisher.subscribe(self.updateProgress, "update")


    def setRange(self, r):
        self.rng = int(r)
        self.progress.SetRange(self.rng)

    def getTitle(self, t):
        self.SetTitle(t)
 
    def updateProgress(self, msg):

        self.count += 1

        if self.count >= int(self.rng-12):
            self.Destroy()
 
        self.progress.SetValue(self.count)
        # self.SetTitle()
        message = str(msg).split('Data:')[1].replace("'", '').replace(']','').replace('[','')
        self.text.SetLabel("Completed: " + message)
