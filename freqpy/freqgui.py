from extimports import *
from importfiles import ImportFiles
from labeldata import LabelData
from analyze import Analyze

class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="FreqPy", size=(500, 800))

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
        file_choices = "TXT (*.txt)|*.txt"
        dialog = wx.FileDialog(
        self, 
        message="Import file...",
        defaultDir=os.getcwd(),
        defaultFile="",
        wildcard=file_choices,
        style=wx.OPEN|wx.MULTIPLE)
        def multipaths(dialog):
            for d in dialog.GetPaths():
                yield d
        if dialog.ShowModal() == wx.ID_OK:
            for each in multipaths(dialog):
                self.import_files.listCtrl.Append([each.split('\\')[-1], 
                                               self.import_files.state])
                self.label_data.listCtrl.Append([each, 
                                             self.import_files.state])
                if self.import_files.state == 'Neural':
                    self.import_files.neurons.append(each)
                    self.label_data.load_data(self.import_files.neurons)
                    self.analyze.load_data(self.import_files.neurons)
                elif self.import_files.state == 'Condition':
                    self.import_files.conditions.append(each)
                    self.analyze.load_conditions(self.import_files.conditions)
                    self.label_data.load_conditions(self.import_files.conditions)

        dialog.Destroy()

if __name__ == '__main__':
    app = wx.App()
    app.frame = MainFrame()
    app.frame.Show()
    app.MainLoop()
    sys.exit()
