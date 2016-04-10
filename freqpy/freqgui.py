from extimports import *
from importfiles import ImportFiles
from labeldata import LabelData
from analyze import Analyze
from visualize import Visualize

def waveforms():
    waveform_names = {
                  5: 'inf_sine',
                  2: 'CL',
                  3: 'low_sine',
                  1: 'no_sim',
                  4: 'top_sine',
                  6: 'tugs_ol',
                  7: 'other'}
    return list(waveform_names.values())

def init_func(fig, axes, axes2, title_, ax_labels, projected, labels, all_ret=True, color=None, i=None):
    wave_labels = waveforms()
    color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
    centers = list()
    classes = list()
    axes.cla()
    plt.setp(axes.get_xticklabels(), fontsize=4)
    plt.setp(axes.get_yticklabels(), fontsize=4)
    plt.setp(axes.get_zticklabels(), fontsize=4)
    if title_ == 'PCA': div_scale = 2
    elif title_ == 'MDA': div_scale = 1.2
    elif title_ == 'k-Means': div_scale = 2
    elif title_ == 'ICA': div_scale = 2 
    allmin = np.min(projected, 0) / div_scale
    allmax = np.max(projected, 0) / div_scale
    axes.set_xlim3d([allmin[0], allmax[0]])
    axes.set_ylim3d([allmin[1], allmax[1]])
    axes.set_zlim3d([allmin[2], allmax[2]])
    text_label = "Frame #: %d" % int(0)
    frame_no = axes.text2D(0.99, 0.01, text_label,
           verticalalignment='bottom', horizontalalignment='right',
           color='b', fontsize=5, transform=axes.transAxes, animated=False)
    axes2.cla()
    plt.setp(axes2.get_xticklabels(), fontsize=4)
    plt.setp(axes2.get_yticklabels(), fontsize=4)
    if i != None:
        axes2.axvline(i, color='k')
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
        axes2.scatter(idx_where, [label]*len(idx_where),
                      color=color_list[int(label)-1])
        classes.append(curr_class)
        centers.append(center)
    axes.legend(handles=classes,
     scatterpoints=1, ncol=1, fontsize=8, 
     labels=wave_labels, frameon=False, 
     bbox_to_anchor=(1, 1))
    axes.set_title(title_, size=10, y=1.0)
    axes.set_xlabel(ax_labels[0],size=5)
    axes.set_ylabel(ax_labels[1],size=5)
    axes.set_zlabel(ax_labels[2],size=5)
    if all_ret:
        return centers, classes, frame_no
    return centers, classes

def opener(names):
    df = dict()
    for name in names:
        if name == '_tmp.txt':
            with open(name, 'r') as nf:
                df[name] = [line for line in nf]
        elif name == 'pdat_labels.txt':
            df[name] = np.loadtxt(name)
    of = dict()
    for k, it in df.items():
        if k == '_tmp.txt':
            of['title'] = it[0].split(':')[1].replace('\n','')
            of['axes_labels'] = eval(it[1].split(':')[1].replace('\n', ''))
            of['out_name'] = it[2].split(':')[1].replace('\n', '')
        elif 'labels' in k:
            of['labels'] = it
    if of['title'] == 'PCA':
        os.chdir('Data')
        of['data'] = np.loadtxt([fi for fi in os.listdir('.') if 'normalized_freq.txt' in fi][0])
        os.chdir('..')
        pca = PCA(n_components=3)
        of['projected'] = pca.fit_transform(of['data'].T)
    elif of['title'] == 'ICA':
        os.chdir('Data')
        of['data'] = np.loadtxt([fi for fi in os.listdir('.') if 'normalized_freq.txt' in fi][0])
        os.chdir('..')
        ica = FastICA(n_components=3)
        of['projected'] = ica.fit_transform(of['data'].T)
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

def save_anim():
    try:
        os.mkdir('./tmp')
    except Exception:
        pass
    waveform_list = waveforms()
    color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
    input_dict = opener(['_tmp.txt', 'pdat_labels.txt'])
    out_movie = input_dict['out_name']
    projected = input_dict['projected']
    labels = input_dict['labels']
    plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[7, 1])
    axes = plt.subplot(gs[0], projection='3d', frame_on=False)
    axes2 = plt.subplot(gs[1], frame_on=False) # waveform
    axes2.set_xlim([0, len(labels)])
    axes2.set_ylim([0, 1.05])
    axes2.cla()
    axes2.axvline(0, color='k')
    plot_args = (fig, axes, axes2,
                 input_dict['title'], input_dict['axes_labels'], 
                 input_dict['projected'], input_dict['labels'])
    centers, classes, frame_no = init_func(*plot_args)
    range_curr = 4
    if plot_args[3] == 'PCA':
        scale_fact = 2.7
    elif plot_args[3] == 'MDA':
        scale_fact = 1.1
    if plot_args[3] == 'k-Means':
        scale_fact = 2.7
    elif plot_args[3] == 'ICA':
        scale_fact = 2.7
    total_range = np.arange(1, len(labels)-range_curr-1)
    filenames = list()
    last_pts = [projected[range_curr:range_curr+1, 0], 
                    projected[range_curr:range_curr+1, 1], 
                    projected[range_curr:range_curr+1, 2]]
    last_color = color_list[0]
    os.chdir('./tmp')
    for i in total_range:
        print(i)
        color = color_list[int(labels[i])-1]
        centers, classes = init_func(*plot_args, all_ret=False, color=color, i=i)
        center = centers[int(labels[i]-1)]
        axes.view_init(elev=30., azim=i)
        curr_projected = projected[i-range_curr:i+range_curr, :]
        curr_label = [color_list[int(cc)-1] for cc in labels[i-range_curr:i+range_curr]]
        try:
            x = curr_projected[:, 0] / scale_fact
        except Exception as E:
            print(E)
        y = curr_projected[:, 1] / scale_fact
        z = curr_projected[:, 2] / scale_fact
        axes.scatter(x, y, z, marker='o', s=10, c=curr_label, alpha=0.8, label=unicode(i))
        last_arr = np.asarray(last_pts)
        curr_xyz = np.asarray([x, y, z])
        for start, end in zip(last_arr.T, curr_xyz.T):
            axes.plot([start[0], end[0]], 
                           [start[1], end[1]], 
                           zs=[start[2], end[2]], 
                           lw=1.0, color=color, label=color, alpha=1.0)
        last_color = color
        last_pts = [x, y, z]
        fig.canvas.draw()
        filename = '__frame%03d.png' % int(i)
        fig.savefig(filename, dpi=100)
        filenames.append(filename)
    subprocess.call('ffmpeg -framerate 20 -i __frame%03d.png -r ntsc ' + out_movie, shell=True)
    for fi in filenames:
        os.remove(fi)
    os.chdir('..')

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
        self.visualize.save_button.Bind(wx.EVT_BUTTON, self.open_vis_thread)
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

    def open_vis_thread(self, event):
        title_ = self.visualize.title_
        ax_labels = self.visualize.ax_labels
        labels = self.visualize.labels
        out_movie = self.visualize.out_movie
        plot_args = (title_, ax_labels, out_movie)
        with open('_tmp.txt', 'w') as tf:
            tf.write('title:' + title_ +'\n')
            tf.write('ax_labels:' + str(ax_labels) +'\n') 
            tf.write('outmoviename:' + out_movie + '\n')
        pool = Pool(processes=cpu_count()*2)
        pool.apply_async(save_anim)
        pool.close()

    def check_page(self, event):
        if self.nb.GetPageText(self.nb.GetSelection()) == "Visualize":
            self.visualize.vis_selected = True
            self.visualize.init_viz()
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
        # os.chdir('..')
        dialog.Destroy()

if __name__ == '__main__':
    app = wx.App()
    app.frame = MainFrame()
    app.frame.Show()
    app.MainLoop()
    sys.exit()