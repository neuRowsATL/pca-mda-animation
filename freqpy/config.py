import wx
import json
import wx.grid as gridlib
import os

class PreferencesDialog(wx.Dialog):
    def __init__(self):
        wx.Dialog.__init__(self, None, wx.ID_ANY, 'Preferences', size=(550,300))

        self.settings = dict()

        self.__create_widgets()
        self.__layout()

    def __create_widgets(self):
        ### PANEL
        self.panel = wx.Panel(self)
        
        ### GRID
        self.grid = gridlib.Grid(self.panel)

        ### OK BUTTON
        self.ok_button = wx.Button(self.panel, -1, "Ok", style=wx.BU_EXACTFIT)
        self.ok_button.Bind(wx.EVT_BUTTON, self.OnOk)

    def __layout(self):
        ### VBOX
        self.vbox = wx.BoxSizer(wx.VERTICAL)

        ### STATIC BOX SIZER (PANEL)
        self.sb = wx.StaticBox(self.panel, label='FreqPy Settings')
        self.sbs = wx.StaticBoxSizer(self.sb, orient=wx.VERTICAL)

        ### ADD WIDGETS TO SBS
        self.sbs.Add(self.grid, -1, flag=wx.EXPAND|wx.RIGHT)
        self.sbs.Add(self.ok_button, -1, flag=wx.EXPAND)

        self.panel.SetSizer(self.sbs)

        ### ADD PANEL TO VBOX
        self.vbox.Add(self.panel, proportion=1, flag=wx.ALL|wx.EXPAND, border=5)

        self.SetSizer(self.vbox)

    def __populate_grid(self):
        self.grid.CreateGrid(numRows=len(self.settings.keys()), numCols=2)
        self.grid.SetColLabelValue(0, "Setting")
        self.grid.SetColLabelValue(1, "Value")
        for ix in range(len(self.settings.items())):
            self.grid.SetCellValue(ix, 0, self.settings.keys()[ix])
            self.grid.SetReadOnly(ix, 0, True)
            try:
                self.grid.SetCellValue(ix, 1, self.settings.values()[ix])
            except TypeError:
                self.grid.SetCellValue(ix, 1, str(self.settings.values()[ix]))

    def get_settings(self, settings):
        self.settings = settings
        self.__populate_grid()

    def OnOk(self, evt):
        self.update_settings()
        self.save_settings(os.path.join(os.getcwd(), os.path.normpath('SETTINGS/SETTINGS.json')))
        self.Close(True)

    def update_settings(self):
        for ix in range(len(self.settings.items())):
            key = self.grid.GetCellValue(ix, 0)
            self.settings[key] = self.grid.GetCellValue(ix, 1)
        return self.settings

    def save_settings(self, sf):
        with open(sf, 'w') as sfw:
            json.dump(self.settings, sfw)
        return 0

