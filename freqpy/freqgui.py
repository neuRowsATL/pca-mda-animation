from extimports import *
from importfiles import ImportFiles
from labeldata import LabelData
from analyze import Analyze
from visualize import Visualize
from clusterize import Clusterize
from compareize import Compareize
from formatize import FormatFileNames
from readme import ReadMe
if sys.platform[0:3] == "win":
    from prog_diag import MyProgressDialog


def opener(names):
    df = dict()
    for name in names:
        if name == '_tmp.txt':
            with open(name, 'r') as nf:
                df[name] = [line for line in nf]
        elif 'labels' in name:# and 'inlier' in name:
            df[name] = np.loadtxt(name)
    of = dict()
    for k, it in df.items():
        if k == '_tmp.txt':
            of['title'] = it[0].split(':')[1].replace('\n','')
            of['axes_labels'] = eval(it[1].split(':')[1].replace('\n', ''))
            of['out_name'] = it[2].split(':')[1].replace('\n', '')
            of['dpi'] = int(it[3].split(':')[1].replace('\n', ''))
        elif 'labels' in k:
            of['labels'] = it
    if of['title'] == 'PCA':
        os.chdir('Data')
        of['data'] = np.loadtxt([fi for fi in os.listdir('.') if 'normalized_freq.txt' in fi][0])
        if of['data'].shape[0] < of['data'].shape[1]: of['data'] = of['data'].T
        os.chdir('..')
        pca = PCA(n_components=3)
        of['projected'] = pca.fit_transform(of['data'])
    elif of['title'] == 'ICA':
        os.chdir('Data')
        of['data'] = np.loadtxt([fi for fi in os.listdir('.') if 'normalized_freq.txt' in fi][0])
        if of['data'].shape[0] < of['data'].shape[1]: of['data'] = of['data'].T
        os.chdir('..')
        ica = FastICA(n_components=3)
        of['projected'] = ica.fit_transform(of['data'])
    elif of['title'] == 'MDA':
        os.chdir('Data')
        of['projected'] = np.loadtxt('_mda_projected.txt')
        of['labels'] = np.loadtxt('_mda_labels.txt')
        os.remove('_mda_labels.txt')
        os.remove('_mda_projected.txt')
        os.chdir('..')
    elif of['title'] == 'K-Means (PCA)':
        os.chdir('Data')
        of['projected'] = np.loadtxt('_kmeans_projected.txt')
        of['labels'] = np.loadtxt('_kmeans_labels.txt')
        os.remove('_kmeans_labels.txt')
        os.remove('_kmeans_projected.txt')
        os.chdir('..')
    os.remove('_tmp.txt')
    return of

def waveforms(folder):
    with open(os.path.join(folder, 'waveform_names.json'), 'r') as wf:
        waveform_names = json.load(wf)
    return list(waveform_names.values())

def init_func(fig, axes, axes2, title_, ax_labels, projected, 
            labels, waveform, data_dir, all_ret=True, color=None, i=None):
    wave_labels = waveforms(data_dir)
    color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
    centers = list()
    classes = list()

    axes.cla()
    plt.setp(axes.get_xticklabels(), fontsize=4)
    plt.setp(axes.get_yticklabels(), fontsize=4)
    plt.setp(axes.get_zticklabels(), fontsize=4)
    allmin = np.min(projected, 0)
    allmax = np.max(projected, 0)
    axes.set_xlim3d([allmin[0], allmax[0]])
    axes.set_ylim3d([allmin[1], allmax[1]])
    axes.set_zlim3d([allmin[2], allmax[2]])

    text_label = "Frame #: %d" % int(0)
    frame_no = axes.text2D(0., -0.3, text_label,
           verticalalignment='bottom', horizontalalignment='left',
           color='b', fontsize=5, transform=axes.transAxes, animated=False)

    axes2.cla()
    plt.setp(axes2.get_xticklabels(), fontsize=4)
    plt.setp(axes2.get_yticklabels(), fontsize=4)
    axes2.set_xticks(np.arange(0, len(labels), 100))
    axes2.set_xticks(np.arange(0, len(labels), 10), minor=True)
    axes2.plot(waveform, color='k', lw=0.5)

    if i != None:
        axes2.axvline(i, color='r')
        frame_no.set_text("Frame #: %d" % int(i))
    for label in set(labels):
        class_proj = projected[labels==label, :]
        center = np.mean(class_proj, 0)
        if color != None:
            if color == color_list[int(label)-1]:
                aa = 1.0
            else:
                aa = 0.25
        elif color == None: aa = 0.25
        curr_class=axes.scatter(center[0], center[1], center[2], 
              marker='o', s=50, edgecolor='k', 
              c=color_list[int(label)-1],
              label=color_list[int(label)-1], alpha=aa)
        idx_where = np.where(labels == label)[0]
        classes.append(curr_class)
        centers.append(center)
    
    if 'K-Means (PCA)' != title_:
        axes.legend(handles=classes, loc=8,
         scatterpoints=1, ncol=len(set(labels)), fontsize=4.5, 
         labels=wave_labels, frameon=False, 
         bbox_to_anchor=(0., -0.46, 1.0, 0.09), mode='expand',
         borderaxespad=0., borderpad=0., labelspacing=5,
         columnspacing=5, handletextpad=0.
         )
    
    axes.set_title(title_, size=10, y=1.0)
    axes.set_xlabel(ax_labels[0],size=5)
    axes.set_ylabel(ax_labels[1],size=5)
    axes.set_zlabel(ax_labels[2],size=5)
    axes.labelpad = 0
    axes.OFFSETTEXTPAD = 0

    axes2.set_title('Waveform', size=5)
    if all_ret:
        return centers, classes, frame_no
    return centers, classes

def save_anim(data_dir, export_dir):
    t0 = time.time()
    try:
        os.mkdir(export_dir+'tmp')
    except Exception:
        pass
    waveform_list = waveforms(data_dir)
    color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
    input_dict = opener(['_tmp.txt', data_dir+'waveform.txt', data_dir+'pdat_labels.txt']) # data_dir+'inlier_labels.txt'
    out_movie = input_dict['out_name']
    projected = input_dict['projected']

    # interpolation (bezier)
    projected = bezier(projected, res=1000)

    labels = input_dict['labels']
    waveform = np.loadtxt(data_dir+'waveform.txt')
    dpi = int(input_dict['dpi'])

    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    gs = gridspec.GridSpec(2, 1, height_ratios=[7, 1])
    gs.update(hspace=1)
    axes = plt.subplot(gs[0], projection='3d', frame_on=True)
    axes2 = plt.subplot(gs[1], frame_on=True) # waveform

    axes2.cla()
    axes2.set_xticks(np.arange(0, len(labels), 100))
    axes2.set_xticks(np.arange(0, len(labels), 10), minor=True)
    axes2.plot(waveform, color='k')
    axes2.axvline(0, color='r')
    plot_args = (fig, axes, axes2,
                 input_dict['title'], input_dict['axes_labels'], 
                 input_dict['projected'], input_dict['labels'],
                 waveform, data_dir)
    centers, classes, frame_no = init_func(*plot_args)

    range_curr = 10
    total_range = np.arange(1, len(projected)-range_curr-1)

    last_pts = [projected[range_curr:range_curr+1, 0], 
                    projected[range_curr:range_curr+1, 1], 
                    projected[range_curr:range_curr+1, 2]]
    last_color = color_list[0]

    os.chdir(export_dir+'tmp')
    filenames = list()
    
    fig.canvas.blit()
    for i in total_range:
        color = color_list[int(labels[i])-1]
        centers, classes = init_func(*plot_args, all_ret=False, color=color, i=i)
        center = centers[int(labels[i]-1)]
        axes.view_init(elev=30., azim=i)
        curr_projected = projected[i-range_curr:i+range_curr, :]
        curr_label = [color_list[int(cc)-1] for cc in labels[i-range_curr:i+range_curr]]
        try:
            x = curr_projected[:, 0]
            y = curr_projected[:, 1]
            z = curr_projected[:, 2]
        except Exception as E:
            print(E)

        last_arr = np.asarray(last_pts)
        curr_xyz = np.asarray([x, y, z])
        acoef = 0.1
        scoef = 0.5
        for start, end in zip(last_arr.T, curr_xyz.T):
            axes.plot([start[0], end[0]], 
                      [start[1], end[1]], 
                      zs=[start[2], end[2]], 
                      lw=1.0, color=color, label=color, alpha=acoef,
                      markersize=scoef,
                      marker='o')
            scoef += 0.15
            acoef += 0.1

        last_color = color
        last_pts = [x, y, z]
        fig.canvas.draw()
        filename = '__frame%03d.png' % int(i)
        fig.savefig(filename, dpi='figure')
        filenames.append(filename)
        if sys.platform[0:3] == "win":
            wx.CallAfter(Publisher.sendMessage, "update", msg='{0} of {1}'.format(i, len(total_range)))

    crf = 30
    reso = '1280x720'
    if dpi == 150: 
        crf = 25
        reso = '2560x1440'
    elif dpi == 200:
        crf = 20
        reso = '5120x2880'
    if sys.platform[0:3] == 'win':
        command = 'ffmpeg -framerate 20 -i __frame%03d.png -s:v ' + reso + ' -c:v libx264 ' +\
                  '-crf ' + str(crf) + ' -tune animation -pix_fmt yuv420p ' + out_movie
    elif sys.platform[0:3] == 'dar':
        command = 'ffmpeg -framerate 20 -i __frame%03d.png -s:v ' + reso + ' -c:v libx264 -crf ' + str(crf) +\
                  ' ' + out_movie
    return_code = subprocess.call(command, shell=True)
    return filenames

class SaveThread(Thread):
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.start()

    def run(self):
        self.func(self.args[0], self.args[1])


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

        OnLoad, EVT_ON_LOAD = wx.lib.newevent.NewEvent()
        wx.PostEvent(self, OnLoad(attr1=True))
        self.Bind(EVT_ON_LOAD, self.on_add_file)

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