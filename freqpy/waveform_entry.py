import wx
import wx.grid as gridlib
from utils import get_waveform_names

class WaveformEntry(wx.Dialog):
    def __init__(self, *args, **kw):
        super(WaveformEntry, self).__init__(*args, **kw)
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        sb = wx.StaticBox(panel, label='Enter Label Names')

        sbs = wx.StaticBoxSizer(sb, orient=wx.VERTICAL)

        self.open_button = wx.Button(panel, -1, "Open JSON", style=wx.BU_EXACTFIT)
        self.open_button.Bind(wx.EVT_BUTTON, self.open_dialog)

        self.ok_button = wx.Button(panel, -1, "Ok", style=wx.BU_EXACTFIT)
        self.ok_button.Bind(wx.EVT_BUTTON, self.OnOk)

        sbs.Add(self.open_button)

        self.grid = gridlib.Grid(panel)

        sbs.Add(self.grid)

        sbs.Add(self.ok_button, -1, flag=wx.EXPAND)

        panel.SetSizer(sbs)

        vbox.Add(panel, proportion=1, flag=wx.ALL|wx.EXPAND, border=5)
        self.SetSizer(vbox)
        
    def open_dialog(self, event):
        dialog = wx.FileDialog(self,
                      message="Select Labels JSON File",
                      style=wx.OPEN|wx.MULTIPLE
                    )
        if dialog.ShowModal() == wx.ID_OK:
            p = dialog.GetPath()
            dialog.Destroy()
            self.wv_file = p
        else:
            self.wv_file = None
        self.file2grid()

    def file2grid(self):
        if self.wv_file is not None:
            self.wv_names = get_waveform_names(self.wv_file)
            for wi, wn in enumerate(self.wv_names):
                self.grid.SetCellValue(wi, 1, wn)

    def getCellVals(self):
        vals = list()
        for r in range(self.grid.GetNumberRows()):
            vals.append(self.grid.GetCellValue(r, 1))
        return vals

    def SetNumRows(self, nr_rows=None):
        self.nr_rows = nr_rows
        self.grid.CreateGrid(self.nr_rows, 2)
        self.grid.SetColLabelValue(0, "Class #")
        self.grid.SetColLabelValue(1, "Class Name")
        for r in range(self.nr_rows):
            self.grid.SetCellValue(r, 0, str(r+1))

    def OnOk(self, evt):
        self.Close(True)
