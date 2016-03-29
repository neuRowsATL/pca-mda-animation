from extimports import *
from arrow3d import Arrow3D
from mda import MDA

class Visualize(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        self.colors = {'k': (0.0, 0.0, 0.0), 'b': (0.0, 0.0, 1.0), 'm': (0.75, 0, 0.75), 
                       'r': (1.0, 0.0, 0.0), 'y': (0.75, 0.75, 0), 'w': (1.0, 1.0, 1.0), 
                       'g': (0.0, 0.5, 0.0), 'c': (0.0, 0.75, 0.75)}
        self.vis_selected = False
        self.t = 0
        self.dpi = 200
        self.fig = Figure((5.0, 5.0), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        # self.play_button = wx.Button(self, -1, "Play Movie")
        self.data_arr = dict()
        self.cond_arr = dict()
        self.lb_arr = list()
        self.lb_condarr = list()
        self.neurons = list()
        self.conditions = list()
        self.files = list()
        self.create_listbox()
        self.save_button = wx.Button(self, -1, "Export Visualization as .mp4")
        # self.save_button.Disable()
        # self.Bind(wx.EVT_BUTTON, self.save_anim, self.save_button)
        # self.Bind(wx.EVT_BUTTON, self.play, self.play_button)
        self.__do_layout()

    def create_listbox(self):
        sampleList = list()
        condList = list()
        for ii, k in enumerate(self.data_arr.keys()):
            if ii > 0 and self.data_arr.keys()[ii-1].split('\\')[-1].split('_')[0] != self.data_arr.keys()[ii].split('\\')[-1].split('_')[0]:
                sampleList.append(self.data_arr.keys()[ii].split('\\')[-1].split('_')[0])
            if ii == 0:
                sampleList.append(self.data_arr.keys()[ii].split('\\')[-1].split('_')[0])
        for ii, k in enumerate(self.cond_arr.keys()):
            if ii > 0 and self.cond_arr.keys()[ii-1].split('\\')[-1].split('_')[0] != self.cond_arr.keys()[ii].split('\\')[-1].split('_')[0]:
                condList.append(self.cond_arr.keys()[ii].split('\\')[-1].split('_')[0])
            if ii == 0:
                condList.append(self.cond_arr.keys()[ii].split('\\')[-1].split('_')[0])
        self.lbtitle = wx.StaticText(self, -1, "Choose Frequency Data:", (80, 10))
        self.lb = wx.Choice(self, -1, (80, 50), wx.DefaultSize, sampleList)
        self.condtitle = wx.StaticText(self, -1, "Choose Condition File:", (80, 10))
        self.lb_cond = wx.Choice(self, -1, (80, 50), wx.DefaultSize, condList)
        self.alg_title = wx.StaticText(self, -1, "Analyze with...:", (80, 10))
        self.alg_choice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, ["PCA", "MDA", "k-Means"])
        self.alg_choice.SetSelection(0)
        self.Bind(wx.EVT_CHOICE, self.plot_selected, self.alg_choice) # Algorithm selection

    def to_freq(self, data):
        nr_pts = 1e3
        vals = np.fromiter(itertools.chain.from_iterable(data.values()),dtype=np.float)
        if len(vals) > 0:
            time_space = np.linspace(min(vals), max(vals), nr_pts)
            delta = time_space[1] - time_space[0]
            time_space = np.insert(time_space, 0, time_space[0] - delta)
            time_space = np.insert(time_space, -1, time_space[-1] + delta)
            freq = np.zeros((max(data.keys())+1, nr_pts))
            for neuron, datum in data.items():
                for ii in np.arange(nr_pts):
                    count = len(datum[np.where((datum < time_space[ii]) & (datum > time_space[ii - 1]))])
                    freq[neuron, ii] = np.divide(count, delta)
            freq = np.divide(freq - np.tile(np.mean(freq), (1, len(freq.T))), 
                             np.tile(np.std(freq), (1, len(freq.T))))
            freq = (1.000 + np.tanh(freq)) / 2.000
            return freq

    def load_data(self, filenames):
        data = dict()
        if len(filenames) > 0:
            for ii, filename in enumerate(filenames):
                if ii == 0:
                    tagname = filename.split('_')[0]
                if filenames[ii-1].split('_')[0] == filenames[ii].split('_')[0]:
                    datum = np.loadtxt(filename,skiprows=2)
                    data.update({ii: datum})
            freq = self.to_freq(data)
            self.data_arr.update({tagname: freq})
            np.savetxt(tagname + "_normalized_freq.txt", freq)
            if self.t == 0:
                self.init_plot()
                self.t += 1
            for ii, k in enumerate(self.data_arr.keys()):
                if ii > 0 and self.data_arr.keys()[ii-1].split('_')[0] != self.data_arr.keys()[ii].split('_')[0]:
                    self.lb.InsertItems([self.data_arr.keys()[ii].split('\\')[-1].split('_')[0]],0)
                    self.lb.SetSelection(0)
                    self.lb_arr.append(self.data_arr.keys()[ii].split('\\')[-1].split('_')[0]) 
                if ii == 0:
                    self.lb.InsertItems([self.data_arr.keys()[ii].split('\\')[-1].split('_')[0]],0)
                    self.lb.SetSelection(0)
                    self.lb_arr.append(self.data_arr.keys()[ii].split('\\')[-1].split('_')[0])

    def load_conditions(self, filenames):
        for filename in filenames:
            conds = np.loadtxt(filename)
            self.cond_arr.update({filename: conds})
        for ii, k in enumerate(self.cond_arr.keys()):
            if ii > 0 and self.cond_arr.keys()[ii-1] != self.cond_arr.keys()[ii]:
                self.lb_cond.InsertItems([self.cond_arr.keys()[ii].split('\\')[-1]],0)
                self.lb_cond.SetSelection(0)
                self.lb_condarr.append(self.cond_arr.keys()[ii].split('\\')[-1]) 
            if ii == 0:
                self.lb_cond.InsertItems([self.cond_arr.keys()[ii].split('\\')[-1]],0)
                self.lb_cond.SetSelection(0)
                self.lb_condarr.append(self.cond_arr.keys()[ii].split('\\')[-1])

    def get_current(self, keyname, t):
        init_arr = dict()
        if t == 'Data':
            for ii, _ in enumerate(self.data_arr.keys()):
                if self.data_arr.keys()[ii].split('\\')[-1].split('_')[0] == keyname:
                    init_arr = self.data_arr[_]
                    break
        elif t == 'Cond':
            for ii, _ in enumerate(self.cond_arr.keys()):
                if self.cond_arr.keys()[ii].split('\\')[-1] == keyname:
                    init_arr = self.cond_arr[_]
                    break
        return init_arr

    def get_selection(self, box, t):
        selected = [box.GetString(box.GetSelection())]
        selarr = dict()
        if len(selected) == 1:
            sel = selected[0]
            selarr = self.get_current(sel, t=t)
        return selarr

    def plot_selected(self, event):
        self.axes.cla()
        self.axes.set_axis_bgcolor('white')
        self.axes.set_title('Cluster Analysis', size=10)
        self.axes.set_xlabel('PC1',size=5)
        self.axes.set_ylabel('PC2',size=5)
        self.axes.set_zlabel('PC3',size=5)
        plt.setp(self.axes.get_xticklabels(), fontsize=4)
        plt.setp(self.axes.get_yticklabels(), fontsize=4)
        plt.setp(self.axes.get_zticklabels(), fontsize=4)
        selected_alg = self.alg_choice.GetString(self.alg_choice.GetSelection())
        selected_dat = self.get_selection(self.lb, t='Data')
        selected_labels = self.get_selection(self.lb_cond, t='Cond')
        if len(self.cond_arr.keys()) < 1 and selected_alg in ['PCA', 'MDA']:
            print("To use MDA or PCA, please select both frequency and labels.")
            return
        if len(self.cond_arr.keys()) > 0 and len(self.data_arr.keys()) > 0 and selected_alg in ['PCA', 'MDA']:
            if selected_alg == 'PCA':
                self.pca_selected(selected_dat, selected_labels)
            elif selected_alg == 'MDA':
                self.mda_selected(selected_dat, selected_labels)
        elif len(self.data_arr.keys()) > 0 and len(self.cond_arr.keys()) < 1 and selected_alg == 'k-Means':
            self.kmeans_selected(selected_dat)
        elif len(self.cond_arr.keys()) > 0 and len(self.data_arr.keys()) > 0 and selected_alg == 'k-Means':
            self.kmeans_selected(selected_dat, labels=selected_labels)

    def init_plot(self):
        self.axes = self.fig.add_axes((0, 0, 1, 1), projection='3d')
        self.axes.set_axis_bgcolor('white')
        self.axes.set_title('Cluster Analysis', size=10)
        self.axes.set_xlabel('PC1',size=5)
        self.axes.set_ylabel('PC2',size=5)
        self.axes.set_zlabel('PC3',size=5)
        plt.setp(self.axes.get_xticklabels(), fontsize=4)
        plt.setp(self.axes.get_yticklabels(), fontsize=4)
        plt.setp(self.axes.get_zticklabels(), fontsize=4)
        try:
            selected_dat = self.data_arr[self.data_arr.keys()[0]]
            selected_labels = self.cond_arr[self.cond_arr.keys()[0]]
            self.pca_selected(selected_dat, selected_labels)
        except IndexError:
            return

    def init_func(self):
        self.axes.cla()
        self.axes.set_xlabel('PC1',size=5)
        self.axes.set_ylabel('PC2',size=5)
        self.axes.set_zlabel('PC3',size=5)
        plt.setp(self.axes.get_xticklabels(), fontsize=4)
        plt.setp(self.axes.get_yticklabels(), fontsize=4)
        plt.setp(self.axes.get_zticklabels(), fontsize=4)
        for label in set(self.labels):
            color = self.color_list[int(label)-1]
            ell_array = self.projected[self.labels==label, :]
            center, radii, rotation = EllipsoidTool().getMinVolEllipse(ell_array)
            scatter_center = self.axes.scatter(center[0], center[1], center[2], 
                                               marker='o', c=color, s=50, label=color)
            self.axes.add_artist(scatter_center)
            self.pca_centers.append(center)
            if len(self.legend_hands) < len(set(self.labels)):
                self.legend_hands.append(scatter_center)
        allmin = np.min(np.asarray(self.pca_centers), 0)
        allmax = np.max(np.asarray(self.pca_centers), 0)
        self.axes.set_xlim3d([allmin[0]-0.1, allmax[0]+0.1])
        self.axes.set_ylim3d([allmin[1]-0.1, allmax[1]+0.1])
        self.axes.set_zlim3d([allmin[2]-0.1, allmax[2]+0.1])
        self.axes.legend(handles=self.legend_hands,
             scatterpoints=1, ncol=1, fontsize=8, 
             labels=self.color_list, frameon=False, 
             bbox_to_anchor=(1, 1))
        text_label = 'Frame #: %d' % int(0)
        self.frame_no = self.axes.text2D(0.99, 0.01, text_label,
               verticalalignment='bottom', horizontalalignment='right',
               color='b', fontsize=5, transform=self.axes.transAxes, animated=False)
        self.frame_no = [t for t in self.axes.get_figure().findobj(Text) if t.get_text() == 'Frame #: 0'][0]
        self.axes.add_artist(self.frame_no)
        self.axes.view_init()
        self.canvas.draw()
        self.axes.draw_artist(self.frame_no)

    def pca_selected(self, data, labels):
        self.axes.set_title('PCA', size=10, y=1.0)
        self.labels = labels
        self.last_color = self.color_list[0]
        self.legend_hands = list()
        self.pca_centers = list()
        pca = PCA(n_components=3)
        self.projected = pca.fit_transform(data.T)
        self.init_func()
        self.create_arrows()
        self.last_center = self.pca_centers[0]
        self.fig.canvas.draw()
        self.fig.canvas.blit()
        self.out_movie = 'PCA_Anim.mpg'
        self.anim = animation.FuncAnimation(self.fig, self.update, self.projected.shape[0],
                                            interval=1, repeat=False, blit=False)

    def save_anim(self):
        self.anim.save('PCA_Anim.mp4', fps=30, bitrate=1800, dpi=200)
        self.init_func()

    # def play(self, event):
    #     self.fig.canvas.draw

    def update(self, i):
        def update_3d_arrows(color, i):
            i = int(i)
            for o in self.axes.get_figure().findobj(Arrow3D):
                if int(o.get_label()) != i and self.last_color != color:
                    current_alpha = o.get_alpha()
                    o.set_alpha(0.8*current_alpha)
                elif int(o.get_label()) == i:
                    o.set_alpha(1.0)
        self.axes.view_init(elev=30., azim=i)
        color = self.color_list[int(self.labels[i])-1]
        update_3d_arrows(color, i)
        self.frame_no.set_text('Frame #: %d' % int(i))
        self.last_color = color
        return [arr for arr in self.axes.get_figure().findobj(Arrow3D)] + [self.frame_no]

    def create_arrows(self):
        for i in np.arange(0, len(self.labels)):
            color = self.color_list[int(self.labels[i])-1]
            center = self.pca_centers[int(self.labels[i])-1]
            if self.last_color != color:
                arrow = Arrow3D([self.last_center[0], center[0]], 
                                [self.last_center[1], center[1]],
                                [self.last_center[2], center[2]],
                                mutation_scale=20, lw=1.5, arrowstyle="->",
                                color=color, alpha=0.0, label=i, animated=False)
                self.axes.add_artist(arrow)
            self.last_center = center
            self.last_color = color
        self.init_list = [arr for arr in self.axes.get_figure().findobj(Arrow3D)]

    def mda_selected(self, data, labels):
        mda = MDA(data, labels)
        train_labels, y_train, y_test = mda.fit_transform()
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        self.axes.set_xlabel('D1',size=5)
        self.axes.set_ylabel('D2',size=5)
        self.axes.set_zlabel('D3',size=5)
        for ii in set(labels):
            out = y_train[train_labels==ii, 0:3]
            x = out[:, 0]
            y = out[:, 1]
            z = out[:, 2]
            self.axes.scatter(x, y, z, c=color_list[int(ii-1)], 
                      marker='o', edgecolor='k', label=str(ii))
            center, radii, rotation = EllipsoidTool().getMinVolEllipse(out)
        self.canvas.draw()

    def kmeans_selected(self, selected_data, labels=None):
        X = selected_data
        pca = PCA(n_components=3)
        projected = pca.fit_transform(X.T)
        kmeans = KMeans(n_clusters=len(set(labels)))
        kmeans.fit(projected)
        y_pred = kmeans.labels_
        self.axes.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                              c=y_pred, marker='o', s=30)
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        for ii in set(labels):
            curr_proj = projected[labels==ii, :]
            center, radii, rotation = EllipsoidTool().getMinVolEllipse(curr_proj)
            EllipsoidTool().plotEllipsoid(center, radii, 
                                          rotation, ax=self.axes, plotAxes=False, 
                                          cageColor=color_list[int(ii)-1], cageAlpha=0.7)
        self.canvas.draw()

    def class_creation(self, labels, data):
        classes = dict()
        data = data.T
        for label in range(int(min(labels)), int(max(labels))+1):
            classes[label] = data[labels==label,:]
        return classes

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.canvas, wx.ALIGN_CENTER|wx.GROW)
        # sizer_1.Add(self.play_button, 0, wx.ALIGN_CENTER)
        sizer_1.Add(self.save_button, 0, wx.ALIGN_CENTER)
        sizer_1.Add(self.lbtitle, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.lb, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        sizer_1.Add(self.condtitle, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.lb_cond, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        sizer_1.Add(self.alg_title, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.alg_choice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()