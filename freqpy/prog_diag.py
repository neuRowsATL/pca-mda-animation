from extimports import *

class MyProgressDialog(wx.Dialog):
    
    def __init__(self):
        """Constructor"""
        wx.Dialog.__init__(self, None, title="Progress")
        self.count = 0
 
        self.progress = wx.Gauge(self, range=988)
 
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.progress, 0, wx.EXPAND)
        self.SetSizer(sizer)
 
        Publisher.subscribe(self.updateProgress, "update")
 
    def updateProgress(self, msg):

        self.count += 1
 
        if self.count >= 988:
            self.Destroy()
 
        self.progress.SetValue(self.count)
        self.SetTitle(msg)
        wx.Yield()