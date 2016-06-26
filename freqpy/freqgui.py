from extimports import *

from importfiles import ImportFiles
from labeldata import LabelData
from analyze import Analyze
from visualize import Visualize
from clusterize import Clusterize
from compareize import Compareize

from waveform_entry import WaveformEntry
from label_entry import LabelEntry
# from config import PreferencesDialog

from viz_save import *

if sys.platform[0:3] == "win":
    from prog_diag import MyProgressDialog

class MainFrame(wx.Frame):
    def __init__(self):

        wx.Frame.__init__(self, None, -1, "FreqPy", wx.DefaultPosition, 
                          size=(800, 750), style=wx.SYSTEM_MENU|wx.CAPTION|wx.CLOSE_BOX)

        ### Get Delimiter for OS
        self.delim = ''
        self.plat = ''
        def do_delims(dir_='.'):
            win_delim = "\\"
            dar_delim = "/"
            if sys.platform[0:3] == "win":
                self.delim = win_delim
                self.plat = 'win'
            elif sys.platform[0:3] == "dar":
                self.delim = dar_delim
                self.plat = 'dar'
            return '.'+self.delim+dir_+self.delim
        do_delims()

        self.data_dir = 'Data'
        self.export_dir = 'OUTPUT'

        if not os.path.isdir(self.export_dir):
            os.mkdir(self.export_dir)

        ### Load Settings
        # self.settings_dir = os.path.normpath(r'SETTINGS\\SETTINGS.json')
        # self.settings = get_settings(os.path.join(os.getcwd(), os.path.normpath(r'SETTINGS\\SETTINGS.json')))

        ### Default Resolution
        # try:
        #     self.resolution = int(self.settings['resolution'])
        # except Exception as re:
        #     print(repr(re))
        self.resolution = int(1e3)

        # ### INPUT AND OUTPUT FOLDERS
        # dirs = [('data', self.settings['data_dir']), 
        #         ('export', self.settings['output_dir'])
        #             ]
        # for d in dirs:
        #     try:
        #         if os.path.isdir(d[1]):
        #             if d[0] == 'data': self.data_dir = self.settings['data_dir']
        #             elif d[0] == 'export': self.export_dir = self.settings['output_dir']
        #         else:
        #             raise IOError('%s does not exist!' % (d,))
        #     except IOError:
        #         self.data_dir = do_delims('Data')
        #         self.export_dir = do_delims('OUTPUT')
        #         fm_dlg = wx.MessageDialog(self,
        #               message="%s was not found." % (d,),
        #               caption="Couldn't Find Directory",
        #               style=wx.OK
        #               )
        #         fm_dlg.ShowModal()
        #         fm_dlg.Destroy()

        ### Load Background Color
        # self.bg_color = self.settings['bg_color']

        ### Data files that will be imported
        self.frequency_data = None
        self.labels = None
        self.waveform_names = None
        self.waveform = None
        self.data_files = None

        ### Auto-import (imports demo data folder)
        # if int(self.settings['auto_demo']) == True:
        #     OnLoad, EVT_ON_LOAD = wx.lib.newevent.NewEvent()
        #     wx.PostEvent(self, OnLoad(attr1=True))
        #     self.Bind(EVT_ON_LOAD, self.on_add_file)

        ###################################################################

        #### Set up menu bars
        self.mbar = wx.MenuBar()
        self.SetMenuBar(self.mbar)

        self.sbar = self.CreateStatusBar()

        #### Set up notebook and pages ###
        p = wx.Panel(self)

        self.nb = wx.Notebook(p)

        self.import_files = ImportFiles(self.nb)
        # self.import_files.bg_color = self.bg_color
        self.import_files.DataButton.Bind(wx.EVT_BUTTON, self.set_data_dir)
        self.import_files.SaveButton.Bind(wx.EVT_BUTTON, self.set_export_dir)
        self.import_files.ResButton.Bind(wx.EVT_BUTTON, self.set_resolution)
        self.import_files.DataImpButton.Bind(wx.EVT_BUTTON, self.import_data)
        self.import_files.LabelsImpButton.Bind(wx.EVT_BUTTON, self.import_labels)
        self.import_files.WaveformImpButton.Bind(wx.EVT_BUTTON, self.import_waveform)

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
        demo_data = wx.MenuItem(open_menu, wx.ID_OPEN, '&Data &Folder\tCtrl+D') # maybe add setting that toggles a button to auto load demo data
        open_menu.AppendItem(demo_data)
        self.Bind(wx.EVT_MENU, self.set_data_dir, demo_data)

        filemenu.AppendMenu(wx.ID_OPEN, 'O&pen...', open_menu)

        # pmi = wx.MenuItem(filemenu, wx.ID_ANY, '&Preferences\tCtrl+P')
        # filemenu.AppendItem(pmi)
        # self.Bind(wx.EVT_MENU, self.settings_dialog, pmi)

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

        self.SetMenuBar(self.mbar)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.ALL|wx.EXPAND)
        p.SetSizerAndFit(sizer)
        self.Layout()

        self.Bind(wx.EVT_MENU, self.OnMenu)

    def distribute_settings(self):
        curr_id = self.nb.GetSelection()
        for pi in range(self.nb.GetPageCount()):
            self.nb.ChangeSelection(pi)
            page = self.nb.GetCurrentPage()
            page.SetBackgroundColour(self.settings['bg_color'])
        self.nb.ChangeSelection(curr_id)
        self.resolution = int(self.settings['resolution'])
        self.data_dir = self.settings['data_dir']
        self.export_dir = self.settings['output_dir']
        self.bg_color = self.settings['bg_color']

    def settings_dialog(self, evt):
        prefs = PreferencesDialog()
        prefs.get_settings(self.settings)
        prefs.ShowModal()
        self.settings = prefs.update_settings()
        prefs.Destroy()
        self.distribute_settings()

    def OpenReadme(self, evt):
        # readme = os.path.join(os.getcwd(), 'Readme'+self.delim+'README.md')
        readme = os.path.normpath("./Readme/README.md")
        if self.plat == 'win':
            os.system("Notepad %s" % (readme,))
        elif self.plat == 'dar':
            os.system("open %s" % (readme,))

    def GoToPage(self, evt):
        self.nb.ChangeSelection(evt.GetId())
        # self.AdjustMenus(evt)

    def OnMenu(self, evt):
        evt.Skip()

    def set_resolution(self, evt):
        res_dlg = wx.TextEntryDialog(self, message="Set Resolution (number of samples)", defaultValue="1000")
        res_dlg.ShowModal()
        self.resolution = int(res_dlg.GetValue())
        res_dlg.Destroy()
        # self.settings['resolution'] = self.resolution

    def set_export_dir(self, event):
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
            # self.settings['output_dir'] = self.export_dir
            # save_settings(self.settings, self.settings_dir)

    def set_data_dir(self, event):
        dialog = wx.DirDialog(self,
                              message="Import Data Directory",
                              style=wx.OPEN|wx.MULTIPLE
                            )
        if dialog.ShowModal() == wx.ID_OK:
            self.data_dir = dialog.GetPath() + self.delim
        dialog.Destroy()

    def import_data(self, evt):
        self.import_files.listCtrl.DeleteAllItems()
        self.frequency_data = None
        fd_dlg = wx.TextEntryDialog(self, 
                    message="Enter the prefix of the spike times data files.\n" +\
                            "Don't include hyphens, numbers, or file extensions.",
                            defaultValue="CBCO")
        fd_dlg.ShowModal()
        dfn = fd_dlg.GetValue()
        fd_dlg.Destroy()
        self.data_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if '.' in f and f.split('.')[1]=='txt' \
                      and dfn in f.split('.')[0]]
        try:
            self.frequency_data = load_data(self.data_files, nr_pts=self.resolution)
            np.savetxt(os.path.join(self.data_dir, '_normalized_freq.txt'), self.frequency_data)
        except IOError:
            fm_dlg = wx.MessageDialog(self,
                      message="We couldn't find files that corresponded to that name.\n" +\
                      "Please refer to the readme for more information.",
                      caption="Couldn't Find Data Files",
                      style=wx.OK
                      )
            fm_dlg.ShowModal()
            fm_dlg.Destroy()

        self.initial_run()


    def import_labels(self, evt):
        def labels_load():
            lbls_diag = LabelEntry(self, size=(550, 400))
            lbls_diag.ShowModal()
            self.labels = lbls_diag.labeller(self.data_files, nr_pts=self.resolution)
            lbls_diag.Destroy()
            np.savetxt(os.path.join(self.data_dir, 'pdat_labels.txt'), self.labels.astype(np.uint8))
        try:
            self.labels = np.loadtxt(os.path.join(self.data_dir, 'pdat_labels.txt'))
            if self.labels.shape[0] != self.resolution: labels_load()
        except IOError:
            labels_load()

        self.initial_run()

    def import_waveform(self, evt):
        ### Waveform Names
        try:
            self.waveform_names = get_waveform_names(self.data_dir)
        except IOError:
            wv_entry = WaveformEntry(self)
            wv_entry.SetNumRows(nr_rows=len(set(self.labels)))
            wv_entry.ShowModal()
            self.waveform_names = wv_entry.getCellVals()
            with open(os.path.join(self.data_dir, 'waveform_names.json'), 'w') as wn:
                json.dump(dict([(wii+1, ww) for wii, ww in enumerate(self.waveform_names)]), wn)
            wv_entry.Destroy()

        ### Waveform data
        waveform_list = [ff for ff in os.listdir(self.data_dir) if ff.split('.')[-1] == 'asc']
        txt_wv_list = [ff for ff in os.listdir(self.data_dir) if ff == 'waveform.txt']

        def waveform_creation():
            waveform_file = os.path.join(self.data_dir, waveform_list[0])
            wv_dlg = wx.TextEntryDialog(self, message="Which trace contains the waveform data?", defaultValue="0")
            wv_dlg.ShowModal()
            trace = int(wv_dlg.GetValue())
            wv_dlg.Destroy()
            self.waveform = waveform_compress(waveform_file, trace=trace, n=self.resolution)
            np.savetxt(os.path.join(self.data_dir, 'waveform.txt'), self.waveform)

        if len(waveform_list) > 0 and len(txt_wv_list) < 1:
            waveform_creation()

        elif len(txt_wv_list) > 0:
            self.waveform = np.loadtxt(os.path.join(self.data_dir, 'waveform.txt'))
            if self.waveform.shape[0] != self.resolution: waveform_creation()

        else:
            wv_m = wx.MessageDialog(self,
                      message="There was an issue importing the waveform files.\n" +\
                      "Try saving as waveform_names.json and waveform.txt in the data folder.",
                      caption="Missing Waveform",
                      style=wx.OK
                      )

        self.initial_run()

    def initial_run(self, evt=None):
        # self.import_files.listCtrl.DeleteAllItems()
        if self.frequency_data is not None:
            for each in self.data_files:
                self.import_files.listCtrl.Append([each.split(self.delim)[-1],'Frequency Data'])

        pages = [self.label_data, self.visualize, self.clusterize,
                 self.compareize, self.analyze]
        if self.frequency_data is not None and self.labels is not None:
            for ix, p in enumerate(pages):
                p.data_dir = self.data_dir
                if ix > 0:
                    p.data = self.frequency_data
                p.labels = self.labels
                p.waveform_names = self.waveform_names
                p.waveform = self.waveform

        if self.data_files is not None:
            self.label_data.data = load_data(self.data_files, full=False)
        
        if self.frequency_data is not None and self.labels is not None and self.waveform is not None:
            self.analyze.init_plot()
            self.label_data.init_plot()
            self.clusterize.plotting()
            self.compareize.plotting()
            if self.visualize.vis_selected:
                self.visualize.init_plot()

    def open_vis_thread(self, event):
        self.visualize.plot_selected()
        title_ = self.visualize.title_
        ax_labels = self.visualize.ax_labels
        labels = self.visualize.labels
        out_movie = self.visualize.out_movie
        dpi = self.visualize.dpi
        labels_name = self.visualize.labels_name
        plot_args = (title_, ax_labels, out_movie)
        with open(os.path.join(self.data_dir, '_tmp.txt'), 'w') as tf:
            tf.write('title:' + title_ +'\n')
            tf.write('ax_labels:' + str(ax_labels) +'\n') 
            tf.write('outmoviename:' + out_movie + '\n')
            tf.write('DPI:' + str(dpi) + '\n')
            tf.write('labels_name:' + labels_name + '\n')
        t0 = time.time()
        if sys.platform[0:3] == "dar":
            filenames = save_anim(self.data_dir, self.export_dir, res=self.resolution) # line for mac
        elif sys.platform[0:3] == "win":
            save_thread = SaveThread(func=save_anim, args=(self.data_dir, self.export_dir, self.resolution))
            dlg = MyProgressDialog()
            dlg.getTitle("Exporting Video: %s" % (os.path.join(self.export_dir, out_movie),))
            dlg.setRange(int(self.resolution-12))
            dlg.ShowModal()
            dlg.Destroy()
        t1 = time.time()
        dial = wx.MessageDialog(None,
                              'Exported Video to: %s' % os.path.join(self.export_dir, out_movie),
                              'Done in %.3f seconds.' % round(t1 - t0, 3),
                               wx.OK)
        dial.ShowModal()
        dial.Destroy()
        if sys.platform[0:3] == "win":
            wx.CallAfter(save_thread.join)
        shutil.rmtree(os.path.join(self.export_dir, 'tmp'))

    def OnClose(self, event):
        self.Close(True)

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