from extimports import *
from importfiles import ImportFiles
from labeldata import LabelData
from analyze import Analyze
from visualize import Visualize
from clusterize import Clusterize
from compareize import Compareize
from waveform_entry import WaveformEntry
from viz_save import *

if sys.platform[0:3] == "win":
    from prog_diag import MyProgressDialog

class MainFrame(wx.Frame):
    def __init__(self):

        wx.Frame.__init__(self, None, -1, "FreqPy", wx.DefaultPosition, 
                          size=(800, 750), style=wx.SYSTEM_MENU|wx.CAPTION|wx.CLOSE_BOX)

        ### Get Delimiter for OS ####
        self.delim = ''
        self.plat = ''
        def do_delims(dir_):
            win_delim = "\\"
            dar_delim = "/"
            if sys.platform[0:3] == "win":
                self.delim = win_delim
                self.plat = 'win'
            elif sys.platform[0:3] == "dar":
                self.delim = dar_delim
                self.plat = 'dar'
            return '.'+self.delim+dir_+self.delim
        self.data_dir = do_delims('Data')
        self.export_dir = do_delims('output_dir')
        self.open_counter = 1

        ### Default Resolution
        self.resolution = 1e3

        ### Data files that will be imported
        self.frequency_data = None
        self.labels = None
        self.waveform_names = None
        self.waveform = None

        ### Auto-import (imports demo data folder)
        # OnLoad, EVT_ON_LOAD = wx.lib.newevent.NewEvent()
        # wx.PostEvent(self, OnLoad(attr1=True))
        # self.Bind(EVT_ON_LOAD, self.on_add_file)

        #### Set up menu bars
        self.mbar = wx.MenuBar()
        self.SetMenuBar(self.mbar)

        self.sbar = self.CreateStatusBar()

        #### Set up notebook and pages ###
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

        self.nb.AddPage(self.import_files, "Initialize")
        self.nb.AddPage(self.label_data, "Categorize")
        self.nb.AddPage(self.analyze, "Analyze")
        self.nb.AddPage(self.visualize, "Visualize")
        self.nb.AddPage(self.clusterize, "Clusterize")
        self.nb.AddPage(self.compareize, "Comparize")

        ### File Menu
        filemenu = wx.Menu()

        open_menu = wx.Menu()
        demo_data = wx.MenuItem(open_menu, wx.ID_OPEN, '&Data&Folder\tCtrl+D')
        open_menu.AppendItem(demo_data)
        self.Bind(wx.EVT_MENU, self.on_add_file, demo_data)
        # open_menu.Append(wx.ID_OPEN, 'Data Folder')

        filemenu.AppendMenu(wx.ID_OPEN, 'O&pen...', open_menu)

        qmi = wx.MenuItem(filemenu, wx.ID_EXIT, '&Quit\tCtrl+W')
        filemenu.AppendItem(qmi)
        self.Bind(wx.EVT_MENU, self.OnClose, qmi)

        self.mbar.Append(filemenu, "&File")

        ### Menu to select tabs
        tabmenu = wx.Menu()
        for pp in range(self.nb.GetPageCount()):
            name = self.nb.GetPageText(pp)

            # sets shortcut key ctrl+number to go to each page.
            tabmenu.Append(pp, "%s\tCtrl+%d" % (name, pp+1),"Go to the %s tab." % (name))
            self.Bind(wx.EVT_MENU, self.GoToPage, id=pp)
        tabmenu.AppendSeparator()
        self.mbar.Append(tabmenu, "&Tabs")

        readme_menu = wx.Menu()
        rmi = wx.MenuItem(readme_menu, wx.ID_ANY, '&Open Readme as Text\tCtrl+R')
        readme_menu.AppendItem(rmi)
        self.Bind(wx.EVT_MENU, self.OpenReadme, rmi)

        self.mbar.Append(readme_menu, "&Readme")

        # m = self.nb.GetCurrentPage().Menu
        # n = self.nb.GetCurrentPage().MenuName
        # self.mbar.Append(m, n)

        self.SetMenuBar(self.mbar)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.ALL|wx.EXPAND)
        p.SetSizerAndFit(sizer)
        self.Layout()

        # self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.AdjustMenus)
        self.Bind(wx.EVT_MENU, self.OnMenu)

        # Disable currently selected tab
        # self.mbar.Enable(self.nb.GetSelection(), False)

    def OpenReadme(self, evt):
        if self.plat == 'win':
            os.system("start "+"README.md")
        elif self.plat == 'dar':
            os.system("open "+"README.md")


    def GoToPage(self, evt):
        self.nb.ChangeSelection(evt.GetId())
        self.AdjustMenus(evt)

    def AdjustMenus(self, evt):
        m = self.nb.GetCurrentPage().Menu
        n = self.nb.GetCurrentPage().MenuName

        self.mbar = self.GetMenuBar()
        self.mbar.Replace(1, m, n)

        for page in range(self.nb.GetPageCount()):
            if page == self.nb.GetSelection():
                self.mbar.Enable(page, False)
            else:
                self.mbar.Enable(page, True)

    def OnMenu(self, evt):
        evt.Skip()

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

            if len(data_files) > 0:
                self.frequency_data = load_data(data_files, nr_pts=self.resolution)
            else:
                fd_dlg = wx.TextEntryDialog(self, 
                                    message="Please enter the name of the neural data files.\n" +\
                                     "Don't include numbers or file extensions.", 
                                    defaultValue="CBCO")
                fd_dlg.ShowModal()
                dfn = fd_dlg.GetValue()
                data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.split('.')[1]=='txt' \
                              and dfn in f.split('.')[0]]
                if len(data_files) > 0:
                    self.frequency_data = load_data(data_files, nr_pts=self.resolution)
                else:
                    fm_dlg = wx.MessageDialog(self,
                                          message="We couldn't find files that corresponded to that name.\n" +\
                                          "Please refer to the readme for more information.",
                                          caption="Couldn't Find Data Files",
                                          style=wx.OK
                                          )
                    fm_dlg.ShowModal()
                    fm_dlg.Destroy()

            try:
                self.labels = np.loadtxt(os.path.join(data_dir, 'pdat_labels.txt'))
            except IOError:
                dialog = wx.FileDialog(self,
                              message="Select Labels File OR Event Times File",
                              style=wx.OPEN|wx.MULTIPLE
                            )
                if dialog.ShowModal() == wx.ID_OK:
                    path = dialog.GetPath()
                    dialog.Destroy()
                    try:
                        self.labels = np.loadtxt(path)
                    except ValueError:
                        self.labels = labeller(data_files=data_files, label_times_file=path)[1]
                        np.savetxt(os.path.join(data_dir, 'pdat_labels.txt'), self.labels)
                        lab_dlg = wx.MessageDialog(self,
                                                  message="Labels saved as pdat_labels.txt",
                                                  caption="Labels Saved.",
                                                  style=wx.OK
                                                  )
                        lab_dlg.ShowModal()
                        lab_dlg.Destroy()
                        pass
                else:
                    self.labels = None
                    lab_dlg = wx.MessageDialog(self,
                                              message="You must have a pdat_labels.txt file in your data folder.\n" +\
                                              "Please refer to the readme for more information",
                                              caption="Missing Labels",
                                              style=wx.OK
                                              )
                    lab_dlg.ShowModal()
                    lab_dlg.Destroy()

            except ValueError:
                self.labels = None
                lab_dlg = wx.MessageDialog(self,
                                          message="That file wasn't in the correct format.\n" +\
                                          "Please refer to the readme for more information.",
                                          caption="Missing Labels",
                                          style=wx.OK
                                          )
                lab_dlg.ShowModal()
                lab_dlg.Destroy()

            try:
                self.waveform_names = get_waveform_names(data_dir)
            except IOError:
                wv_entry = WaveformEntry(self)
                wv_entry.SetNumRows(nr_rows=len(set(self.labels)))
                wv_entry.ShowModal()
                self.waveform_names = wv_entry.getCellVals()
                with open(os.path.join(data_dir, 'waveform_names.json'), 'w') as wn:
                    json.dump(dict([(wii+1, ww) for wii, ww in enumerate(self.waveform_names)]), wn)
                wv_entry.Destroy()
        
            waveform_list = [ff for ff in os.listdir(data_dir) if ff.split('.')[-1] == 'asc']
            txt_wv_list = [ff for ff in os.listdir(data_dir) if ff == 'waveform.txt']

            if len(waveform_list) > 0 and len(txt_wv_list) < 1:
                waveform_file = os.path.join(data_dir, waveform_list[0])
                wv_dlg = wx.TextEntryDialog(self, message="Which trace contains the waveform data?", defaultValue="0")
                wv_dlg.ShowModal()
                trace = int(wv_dlg.GetValue())
                wv_dlg.Destroy()
                self.waveform = waveform_compress(waveform_file, trace=trace, n=self.resolution)
                np.savetxt(os.path.join(data_dir, 'waveform.txt'), self.waveform)
            elif len(txt_wv_list) > 0:
                self.waveform = np.loadtxt(os.path.join(data_dir, 'waveform.txt'))
            elif len(txt_wv_list) == 0 and len(waveform_list) == 0:
                self.waveform = None
                wv_m = wx.MessageDialog(self,
                                      message="You must have a waveform.txt file in your data folder.\n" +\
                                      "Please refer to the readme for more information",
                                      caption="Missing Waveform",
                                      style=wx.OK
                                      )
                return 1

            if len(data_files) > 0:
                for each in data_files:
                    self.import_files.listCtrl.Append([each.split(delim)[-1],'Frequency Data'])

            pages = [self.label_data, self.visualize, self.clusterize,
                     self.compareize, self.analyze]
            if self.frequency_data is not None and self.labels is not None:
                for ix, p in enumerate(pages):
                    p.data_dir = data_dir
                    if ix > 0:
                        p.data = self.frequency_data
                    p.labels = self.labels
                    p.waveform_names = self.waveform_names
                    p.waveform = self.waveform

            if len(data_files) > 0:
                self.label_data.data = load_data(data_files, full=False)
            
            if self.frequency_data is not None:
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