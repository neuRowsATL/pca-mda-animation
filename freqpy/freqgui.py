from extimports import *
from importfiles import ImportFiles
from labeldata import LabelData
from analyze import Analyze

class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="FreqPy", size=(700, 800))

        self.neurons = list()
        self.conditions = list()

        p = wx.Panel(self)
        nb = wx.Notebook(p)

        self.import_files = ImportFiles(nb)
        self.import_files.DataButton.Bind(wx.EVT_BUTTON, self.on_add_file)
        self.label_data = LabelData(nb)
        self.analyze = Analyze(nb)

        nb.AddPage(self.import_files, "Import Files")
        nb.AddPage(self.label_data, "Label Data")
        nb.AddPage(self.analyze, "Analyze")

        sizer = wx.BoxSizer()
        sizer.Add(nb, 1, wx.EXPAND)
        p.SetSizer(sizer)
        self.Layout()

    def on_add_file(self, event):
        dialog = wx.DirDialog(
        self,
        message="Import Data Directory",
        style=wx.OPEN|wx.MULTIPLE)
        if dialog.ShowModal() == wx.ID_OK:
            files_ = [os.path.abspath(dialog.GetPath()+"\\"+ff) for ff in os.listdir(dialog.GetPath())]
            files = [f.split("\\")[-1] for f in files_]
            data_files = [f for f in files if all(fl.isdigit() for fl in f.split('D_')[0]) and f.split('.')[-1]=='txt' \
                          and 'labels' not in f.lower() and f.split('.')[0][-1].isdigit()]
            data_files = [df for df in files_ if df.split('\\')[-1] in data_files]
            label_files = [f for f in files if all(fl.isalpha() for fl in f.split('_')[0]) and f.split('.')[-1]=='txt' \
                           and 'labels' in f.lower()]
            label_files = [lf for lf in files_ if lf.split('\\')[-1] in label_files]
            for each in data_files:
                self.import_files.listCtrl.Append([each.split('\\')[-1],'Frequency Data'])
                self.import_files.neurons.append(each)
            for each in label_files:
                self.import_files.listCtrl.Append([each.split('\\')[-1],'Labels'])
                self.import_files.conditions.append(each)
            self.label_data.load_data(self.import_files.neurons)
            self.analyze.load_data(self.import_files.neurons)
            self.analyze.load_conditions(self.import_files.conditions)
            self.label_data.load_conditions(self.import_files.conditions)
            self.analyze.init_plot()
        dialog.Destroy()

if __name__ == '__main__':
    app = wx.App()
    app.frame = MainFrame()
    app.frame.Show()
    app.MainLoop()
    sys.exit()
