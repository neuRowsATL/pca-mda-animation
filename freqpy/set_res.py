from extimports import *

class ResSlider(wx.Frame):
    def __init__(self):

        wx.Frame.__init__(self, None, title="Set Resolution")
        panel = wx.Panel(self, -1)

        self.val = 1e3

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.titler = wx.StaticText(self, -1, 
        							"Choose data resolution.\n Data must be >= 1000 data points.\n Larger datasets will be resampled.", 
        	                        (80, 10))
        self.sld = wx.Slider(panel, -1, 1000, 500, int(1e4), wx.DefaultPosition, (250, -1),
                              wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS)
        btn1 = wx.Button(panel, 8, 'Set Resolution')
        btn2 = wx.Button(panel, 9, 'Close')

        wx.EVT_BUTTON(self, 8, self.OnAdjust)
        wx.EVT_BUTTON(self, 9, self.OnClose)
        vbox.Add(self.sld, 1, wx.ALIGN_CENTRE)
        hbox.Add(btn1, 1, wx.RIGHT, 10)
        hbox.Add(btn2, 1)
        vbox.Add(hbox, 0, wx.ALIGN_CENTRE | wx.ALL, 20)
        panel.SetSizer(vbox)

    def OnAdjust(self, event):
        val = self.sld.GetValue()
        self.val = val
        self.Close()
        return val

        self.SetSize((val*2, val))
    def OnClose(self, event):
        self.Close()