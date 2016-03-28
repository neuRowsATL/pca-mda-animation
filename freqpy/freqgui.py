from extimports import *
from importfiles import ImportFiles
from labeldata import LabelData
from analyze import Analyze
from visualize import Visualize

# Button definitions
ID_START = wx.NewId()
ID_STOP = wx.NewId()

# Define notification event for thread completion
EVT_RESULT_ID = wx.NewId()

def EVT_RESULT(win, func):
    """Define Result Event."""
    win.Connect(-1, -1, EVT_RESULT_ID, func)

class ResultEvent(wx.PyEvent):
    """Simple event to carry arbitrary result data."""
    def __init__(self, data):
        """Init Result Event."""
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_RESULT_ID)
        self.data = data

# Thread class that executes processing
class WorkerThread(Thread):
    """Worker Thread Class."""
    def __init__(self, notify_window, func, anim_name):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self._notify_window = notify_window
        self._want_abort = 0
        self.func = func
        self.anim_name = anim_name
        # This starts the thread running on creation, but you could
        # also make the GUI thread responsible for calling this
        # self.start()

    def run(self):
        """Run Worker Thread."""
        self.func.save(self.anim_name + '.mp4', fps=12, bitrate=1800, extra_args=['-vcodec', 'libx264'], dpi=100)
        return

    def abort(self):
        """abort worker thread."""
        # Method for use by main thread to signal an abort
        self._want_abort = 1

class MainFrame(wx.Frame):
    def __init__(self):

        wx.Frame.__init__(self, None, title="FreqPy", size=(800, 900))

        # Set up event handler for any worker thread results
        EVT_RESULT(self,self.OnResult)

        # And indicate we don't have a worker thread yet
        self.worker = None

        self.neurons = list()
        self.conditions = list()

        p = wx.Panel(self)
        self.nb = wx.Notebook(p)

        self.import_files = ImportFiles(self.nb)
        self.import_files.DataButton.Bind(wx.EVT_BUTTON, self.on_add_file)
        self.label_data = LabelData(self.nb)
        self.analyze = Analyze(self.nb)
        self.visualize = Visualize(self.nb)

        self.nb.AddPage(self.import_files, "Initialize")
        self.nb.AddPage(self.label_data, "Categorize")
        self.nb.AddPage(self.analyze, "Analyze")
        self.nb.AddPage(self.visualize, "Visualize")
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.check_page)
        self.visualize.save_button.Bind(wx.EVT_BUTTON, self.OnStart)

        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        p.SetSizer(sizer)
        self.Layout()

    def OnStart(self, event):
        """Start Computation."""
        if not self.worker:
            self.worker = WorkerThread(self, self.visualize.anim, self.visualize.anim_name)
            self.worker.start()

    def OnStop(self, event):
        """Stop Computation."""
        # Flag the worker thread to stop if running
        if self.worker:
            self.worker.abort()

    def OnResult(self, event):
        self.worker = None

    def save_anim_run(self):
    	self.visualize.anim.save(self.visualize.anim_name + '.mp4', fps=12, bitrate=1800, extra_args=['-vcodec', 'libx264'], dpi=100)
    	
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

class MainApp(wx.App):
    """Class Main App."""
    def OnInit(self):
        """Init Main App."""
        self.frame = MainFrame()
        self.frame.Show(True)
        self.SetTopWindow(self.frame)
        return True

if __name__ == '__main__':
    app = MainApp(0)
    app.MainLoop()
    sys.exit()
