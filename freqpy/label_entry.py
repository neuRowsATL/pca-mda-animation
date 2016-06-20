import wx
import wx.grid as gridlib
from utils import load_data, timeline
import numpy as np

class LabelEntry(wx.Dialog):
    def __init__(self, *args, **kw):
        super(LabelEntry, self).__init__(*args, **kw)

        fd_dlg = wx.TextEntryDialog(self,
                                    message="Please enter the total number of events (Event On -> Event Off).", 
                                    defaultValue="7")
        fd_dlg.ShowModal()
        self.nr_rows = int(fd_dlg.GetValue())
        fd_dlg.Destroy()

        self.label_list = list()

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        sb = wx.StaticBox(panel, label='Enter Event On / Off Times')

        sbs = wx.StaticBoxSizer(sb, orient=wx.VERTICAL)

        self.open_button = wx.Button(panel, -1, "Open Event Times CSV", style=wx.BU_EXACTFIT)
        self.open_button.Bind(wx.EVT_BUTTON, self.open_dialog)

        self.ok_button = wx.Button(panel, -1, "Ok", style=wx.BU_EXACTFIT)
        self.ok_button.Bind(wx.EVT_BUTTON, self.OnOk)

        sbs.Add(self.open_button, -1)

        self.grid = gridlib.Grid(panel)
        self.grid.CreateGrid(self.nr_rows, 3)
        self.grid.SetColLabelValue(2, "Label #")
        self.grid.SetColLabelValue(0, "Event On")
        self.grid.SetColLabelValue(1, "Event Off")

        sbs.Add(self.grid, -1, flag=wx.EXPAND|wx.RIGHT)

        sbs.Add(self.ok_button, -1, flag=wx.EXPAND)

        panel.SetSizer(sbs)

        vbox.Add(panel, proportion=1, flag=wx.ALL|wx.EXPAND, border=5)
        self.SetSizer(vbox)
        
    def open_dialog(self, event):
        dialog = wx.FileDialog(self,
                      message="Select Event Times CSV",
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
            with open(self.wv_file, 'r') as wvf:
                lines = wvf.readlines()
            if len(lines) == 1:
                lines = lines[0].split('\r')
            for li, line in enumerate(lines):
                if 'LABEL' not in line:
                    line = line.split(',')
                    if len(line) == 3:
                        on_time = line[0]
                        off_time = line[1]
                        label = line[2]
                        try:
                            self.grid.SetCellValue(li, 0, on_time)
                            self.grid.SetCellValue(li, 1, off_time)
                            self.grid.SetCellValue(li, 2, label)
                        except Exception as e:
                            self.grid.AppendRows(1, True)
                            self.grid.SetCellValue(li, 0, on_time)
                            self.grid.SetCellValue(li, 1, off_time)
                            self.grid.SetCellValue(li, 2, label)
                        self.label_list.append(line)
    
    def OnOk(self, evt):
        self.Close(True)

    def labeller(self, data_files, nr_pts=1000):
        freq = load_data(data_files)
        time_space = timeline(load_data(data_files, full=False), nr_pts=nr_pts)
        labels = np.ones((nr_pts, 1))
        for l in self.label_list:
            on_time = float(l[0])
            off_time = float(l[1])
            label = int(l[2])
            labels[np.where((time_space >= on_time) & (time_space <= off_time))] = label
        return labels.ravel().astype(np.uint8)

    def SetRows(self, nr_rows):
        pass

