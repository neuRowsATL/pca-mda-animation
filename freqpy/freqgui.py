from extimports import *
from importfiles import ImportFiles
from labeldata import LabelData
from analyze import Analyze
from visualize import Visualize
from clusterize import Clusterize
from compareize import Compareize
from formatize import FormatFileNames
from readme import ReadMe
from viz_save import *

if sys.platform[0:3] == "win":
    from prog_diag import MyProgressDialog

class MainFrame(wx.Frame):
    def __init__(self):

        wx.Frame.__init__(self, None, title="FreqPy", size=(800, 750))

        self.delim = ''
        def do_delims(dir_):
            win_delim = "\\"
            dar_delim = "/"
            if sys.platform[0:3] == "win":
                self.delim = win_delim
            elif sys.platform[0:3] == "dar":
                self.delim = dar_delim
            return '.'+self.delim+dir_+self.delim

        self.data_dir = do_delims('Data')
        self.export_dir = do_delims('output_dir')
        self.open_counter = 1

        self.resolution = 1e3

        self.frequency_data = None
        self.labels = None
        self.waveform_names = None
        self.waveform = None

        # OnLoad, EVT_ON_LOAD = wx.lib.newevent.NewEvent()
        # wx.PostEvent(self, OnLoad(attr1=True))
        # self.Bind(EVT_ON_LOAD, self.on_add_file)

        p = wx.Panel(self)
        self.nb = wx.Notebook(p)

        self.import_files = ImportFiles(self.nb)
        self.import_files.DataButton.Bind(wx.EVT_BUTTON, self.on_add_file)
        self.import_files.SaveButton.Bind(wx.EVT_BUTTON, self.on_save)

        self.label_data = LabelData(self.nb)

        self.analyze = Analyze(self.nb)

        self.visualize = Visualize(self.nb)
        self.visualize.save_button.Bind(wx.EVT_BUTTON, self.open_vis_thread)

        self.clusterize = Clusterize(self.nb)

        self.compareize = Compareize(self.nb)

        self.formatize = FormatFileNames(self.nb)
        self.formatize.ResButton.Bind(wx.EVT_BUTTON, self.OnAdjust)

        self.readme = ReadMe(self.nb)

        self.nb.AddPage(self.readme, "Dogmatize")
        self.nb.AddPage(self.formatize, "Formatize")
        self.nb.AddPage(self.import_files, "Initialize")
        self.nb.AddPage(self.label_data, "Categorize")
        self.nb.AddPage(self.analyze, "Analyze")
        self.nb.AddPage(self.visualize, "Visualize")
        self.nb.AddPage(self.clusterize, "Clusterize")
        self.nb.AddPage(self.compareize, "Comparize")

        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        p.SetSizer(sizer)
        self.Layout()

    def OnAdjust(self, event):
        val = self.formatize.t1.GetValue()
        if int(val) != self.labels.shape[0]:
            try:
                self.frequency_data, self.labels = labeller(data_freq=to_freq(self.label_data.data, nr_pts=int(val)), 
                         data_list=self.label_data.data,
                         nr_pts=int(val))
            except Exception as e:
                print(e)
                return 1
        self.resolution = int(val)
        self.frequency_data = to_freq(self.label_data.data, nr_pts=self.resolution)
        pages = [self.visualize, self.clusterize,
                self.compareize, self.analyze]
        for ix, p in enumerate(pages):
            p.data = self.frequency_data

    def on_save(self, event):
        dialog = wx.DirDialog(self, message="Select Export Directory", style=wx.OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.export_dir = dialog.GetPath()
            win_delim = "\\"
            dar_delim = "/"
            if sys.platform[0:3] == "win":
                delim = win_delim
            elif sys.platform[0:3] == "dar":
                delim = dar_delim
            self.export_dir = self.export_dir+delim
            self.clusterize.export_dir = self.export_dir
            self.compareize.export_dir = self.export_dir

    def OnClose(self, event):
        self.Destroy()
        self.Close(True)

    def open_vis_thread(self, event):
        self.visualize.plot_selected()
        title_ = self.visualize.title_
        ax_labels = self.visualize.ax_labels
        labels = self.visualize.labels
        out_movie = self.visualize.out_movie
        dpi = self.visualize.dpi
        plot_args = (title_, ax_labels, out_movie)
        with open('_tmp.txt', 'w') as tf:
            tf.write('title:' + title_ +'\n')
            tf.write('ax_labels:' + str(ax_labels) +'\n') 
            tf.write('outmoviename:' + out_movie + '\n')
            tf.write('DPI:' + str(dpi) + '\n')
        t0 = time.time()
        if sys.platform[0:3] == "dar":
            filenames = save_anim(self.data_dir, self.export_dir) # line for mac
        elif sys.platform[0:3] == "win":
            save_thread = SaveThread(func=save_anim, args=(self.data_dir, self.export_dir))
            dlg = MyProgressDialog()
            dlg.ShowModal()
        t1 = time.time()
        dial = wx.MessageDialog(None,
                              'Exported Video to: %s' % os.path.join(self.export_dir+'tmp'+'/', out_movie), 
                              'Done in %.3f seconds.' % round(t1 - t0, 3), 
                               wx.OK)
        if dial.ShowModal() == wx.ID_OK:
            for fi in filenames:
                os.remove(fi)
            os.chdir('..')
        if sys.platform[0:3] == "win":
            wx.CallAfter(save_thread.join())

    def on_add_file(self, event):
        def check_platform():
            win_delim = "\\"
            dar_delim = "/"
            if sys.platform[0:3] == "win":
                delim = win_delim
            elif sys.platform[0:3] == "dar":
                delim = dar_delim
            return delim

        def distribute_path(data_dir):
            files_ = [os.path.join(data_dir, ff) for ff in os.listdir(data_dir)]
            files = [f.split(delim)[-1] for f in files_]
            
            data_files = [f for f in files if all(fl.isdigit() for fl in f.split('D_')[0]) and f.split('.')[-1]=='txt' \
                          and 'labels' not in f.lower() and f.split('.')[0][-1].isdigit()]
            data_files = [df for df in files_ if df.split(delim)[-1] in data_files]

            self.data_dir = data_dir

            self.frequency_data = load_data(data_files, nr_pts=self.resolution)

            self.labels = np.loadtxt(os.path.join(data_dir, 'pdat_labels.txt'))

            self.waveform_names = get_waveform_names(data_dir)
        
            waveform_list = [ff for ff in os.listdir(data_dir) if ff.split('.')[-1] == 'asc']
            txt_wv_list = [ff for ff in os.listdir(data_dir) if ff == 'waveform.txt']

            if len(waveform_list) > 0 and len(txt_wv_list) < 1:
                waveform_file = os.path.join(data_dir, waveform_list[0])
                # self.waveform = average_waveform(np.loadtxt(waveform_file), nr_pts=self.resolution)
                self.waveform = waveform_compress(waveform_file, n=self.resolution)
                np.savetxt(os.path.join(data_dir, 'waveform.txt'), self.waveform)
            else:
                self.waveform = np.loadtxt(os.path.join(data_dir, 'waveform.txt'))

            for each in data_files:
                self.import_files.listCtrl.Append([each.split(delim)[-1],'Frequency Data'])

            pages = [self.label_data, self.visualize, self.clusterize,
                     self.compareize, self.analyze]
            for ix, p in enumerate(pages):
                p.data_dir = data_dir
                if ix > 0:
                    p.data = self.frequency_data
                p.labels = self.labels
                p.waveform_names = self.waveform_names
                p.waveform = self.waveform

            self.label_data.data = load_data(data_files, full=False)
            
            self.analyze.init_plot()
            self.label_data.init_plot()

            self.clusterize.plotting()
            self.compareize.plotting()

            if self.visualize.vis_selected:
                self.visualize.init_plot()

        delim = check_platform()

        try:
            etest = event.attr1
        except AttributeError:
            etest = False

        if etest is True:
            y_n_dialog = wx.MessageDialog(self,
                                          message="Import data from %s?" % self.data_dir,
                                          caption="Auto-Import?",
                                          style=wx.YES_NO|wx.YES_DEFAULT|wx.ICON_QUESTION
                                          )
            if y_n_dialog.ShowModal() == wx.ID_NO:
                etest = False
                return 1

        if etest is False:
            dialog = wx.DirDialog(self,
                                  message="Import Data Directory",
                                  style=wx.OPEN|wx.MULTIPLE
                                )
            if dialog.ShowModal() == wx.ID_OK:
                data_dir = dialog.GetPath() + delim
                distribute_path(data_dir)
                dialog.Destroy()
        else:
            distribute_path(self.data_dir)

def main():
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()