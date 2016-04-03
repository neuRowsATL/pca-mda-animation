from extimports import *
from importfiles import ImportFiles
from labeldata import LabelData
from analyze import Analyze
from visualize import Visualize

class MainFrame(wx.Frame):
    def __init__(self):

        wx.Frame.__init__(self, None, title="FreqPy", size=(800, 800))
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.neurons = list()
        self.conditions = list()

        p = wx.Panel(self)
        self.nb = wx.Notebook(p)

        self.import_files = ImportFiles(self.nb)
        self.import_files.DataButton.Bind(wx.EVT_BUTTON, self.on_add_file)
        self.label_data = LabelData(self.nb)
        self.analyze = Analyze(self.nb)
        self.visualize = Visualize(self.nb)
        self.visualize.save_button.Bind(wx.EVT_BUTTON, self.save_anim_run)
        self.visualize.to_freq = self.analyze.to_freq

        self.nb.AddPage(self.import_files, "Initialize")
        self.nb.AddPage(self.label_data, "Categorize")
        self.nb.AddPage(self.analyze, "Analyze")
        self.nb.AddPage(self.visualize, "Visualize")
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.check_page)

        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        p.SetSizer(sizer)
        self.Layout()

    def OnClose(self, event):
        self.Destroy()

    def open_vis_thread(self):
        range_curr = 3
        total_range = len(self.visualize.labels)-range_curr
        ranges = (np.arange(range_curr + 1, int(total_range / 4)),
                 np.arange(int(total_range / 4), int(total_range / 2)),
                 np.arange(int(total_range / 2), int(3*total_range / 4)),
                 np.arange(int(3*total_range / 4), total_range))
        for rr in ranges:
            self.visualize.save_anim(rr)

    def save_anim_run(self, event):
        self.save_diag = wx.MessageBox("This might take a few minutes. \n\
                                        Click OK to begin exporting.",
                                        'Exporting Video as .mp4')
        self.open_vis_thread()

    def check_page(self, event):
        if self.nb.GetPageText(self.nb.GetSelection()) == "Visualize":
            self.visualize.vis_selected = True
            self.visualize.init_plot()
        elif self.nb.GetPageText(self.nb.GetSelection()) != "Visualize":
            self.visualize.vis_selected = False

    def on_add_file(self, event):
        dialog = wx.DirDialog(
        self,
        message="Import Data Directory",
        style=wx.OPEN|wx.MULTIPLE)
        if dialog.ShowModal() == wx.ID_OK:
            win_delim = "\\"
            dar_delim = "/"
            if sys.platform[0:3] == "win":
                delim = win_delim
            elif sys.platform[0:3] == "dar":
                delim = dar_delim
            files_ = [os.path.abspath(dialog.GetPath()+delim+ff) for ff in os.listdir(dialog.GetPath())]
            files = [f.split(delim)[-1] for f in files_]
            data_files = [f for f in files if all(fl.isdigit() for fl in f.split('D_')[0]) and f.split('.')[-1]=='txt' \
                          and 'labels' not in f.lower() and f.split('.')[0][-1].isdigit()]
            data_files = [df for df in files_ if df.split(delim)[-1] in data_files]
            label_files = [f for f in files if all(fl.isalpha() for fl in f.split('_')[0]) and f.split('.')[-1]=='txt' \
                           and 'labels' in f.lower()]
            label_files = [lf for lf in files_ if lf.split(delim)[-1] in label_files]
            for each in data_files:
                self.import_files.listCtrl.Append([each.split(delim)[-1],'Frequency Data'])
                self.import_files.neurons.append(os.path.normpath(each))
            for each in label_files:
                self.import_files.listCtrl.Append([each.split(delim)[-1],'Labels'])
                self.import_files.conditions.append(os.path.normpath(each))
            self.label_data.load_data(self.import_files.neurons)
            self.label_data.load_conditions(self.import_files.conditions)
            self.analyze.load_data(self.import_files.neurons)
            self.analyze.load_conditions(self.import_files.conditions)
            self.analyze.init_plot()
            self.visualize.load_data(self.import_files.neurons)
            self.visualize.load_conditions(self.import_files.conditions)
            if self.visualize.vis_selected:
                self.visualize.init_plot()
        dialog.Destroy()

if __name__ == '__main__':
    app = wx.App()
    app.frame = MainFrame()
    app.frame.Show()
    app.MainLoop()
    sys.exit()