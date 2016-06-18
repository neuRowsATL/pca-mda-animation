from extimports import *

class MyProgressDialog(wx.Dialog):
    
    def __init__(self):
        """Constructor"""
        wx.Dialog.__init__(self, None, title="Progress")
        self.count = 0

        self.rng = 988
 
        self.progress = wx.Gauge(self, range=self.rng)
 
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.progress, 0, wx.EXPAND)
        self.SetSizer(sizer)
        # pubsubconf.transitionV1ToV3('msg', step=2)
        Publisher.subscribe(self.updateProgress, "update")

    def setRange(self, r):
        self.rng = r
        self.progress = wx.Gauge(self, range=self.rng)
 
    def updateProgress(self, msg):

        self.count += 1
 
        if self.count >= 988:
            self.Destroy()
 
        self.progress.SetValue(self.count)
        self.SetTitle(str(msg).split('Data:')[1])
        wx.Yield()