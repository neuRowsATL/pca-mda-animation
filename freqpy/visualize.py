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
        plt.setp(self.axes.get_xticklabels(), fontsize=4)
        plt.setp(self.axes.get_yticklabels(), fontsize=4)
        plt.setp(self.axes.get_zticklabels(), fontsize=4)
        allmin = np.min(self.projected, 0)
        allmax = np.max(self.projected, 0)
        self.axes.set_xlim3d([allmin[0]/2, allmax[0]/2])
        self.axes.set_ylim3d([allmin[1]/2, allmax[1]/2])
        self.axes.set_zlim3d([allmin[2]/2, allmax[2]/2])
        text_label = 'Frame #: %d' % int(0)
        self.frame_no = self.axes.text2D(0.99, 0.01, text_label,
               verticalalignment='bottom', horizontalalignment='right',
               color='b', fontsize=5, transform=self.axes.transAxes, animated=False)
        self.frame_no = [t for t in self.axes.get_figure().findobj(Text) if t.get_text() == 'Frame #: 0'][0]
        self.axes.add_artist(self.frame_no)
        self.axes.view_init()
        self.fig.canvas.draw()
        self.axes.draw_artist(self.frame_no)

    def pca_selected(self, data, labels):
        self.axes.set_title('PCA', size=10, y=1.0)
        self.axes.set_xlabel('PC1',size=5)
        self.axes.set_ylabel('PC2',size=5)
        self.axes.set_zlabel('PC3',size=5)
        self.labels = labels
        self.last_color = self.color_list[0]
        self.legend_hands = list()
        self.pca_centers = list()
        pca = PCA(n_components=3)
        self.projected = pca.fit_transform(data.T)
        self.init_func()
        self.fig.canvas.draw()
        self.fig.canvas.blit()
        self.out_movie = 'PCA_Anim.mp4'
        self.fig.canvas.draw()

    def save_anim(self):
        range_curr = 5
        centers = list()
        filenames = list()
        classes = list()
        for label in set(self.labels):
            class_proj = self.projected[self.labels==label, :]
            center = np.mean(class_proj, 0)
            curr_class=self.axes.scatter(center[0], center[1], center[2], 
                  marker='o', s=50, edgecolor='k', 
                  c=self.color_list[int(label)-1],
                  label=self.color_list[int(label)-1], alpha=0.2)
            classes.append(curr_class)
            centers.append(center)
        self.axes.legend(handles=classes,
         scatterpoints=1, ncol=1, fontsize=8, 
         labels=self.color_list, frameon=False, 
         bbox_to_anchor=(1, 1))
        self.last_center = centers[0]
        self.last_pts = self.projected[0:6, :]
        self.fig.canvas.blit()
        for i in np.arange(0, len(self.labels)-6):
            self.frame_no.set_text("Frame #: %d" % int(i))
            self.axes.view_init(elev=30., azim=i)
            curr_projected = self.projected[i-range_curr:i+range_curr, :]
            curr_label = [self.color_list[int(cc)-1] for cc in self.labels[i-range_curr:i+range_curr]]
            x = curr_projected[:, 0] / 4
            y = curr_projected[:, 1] / 4
            z = curr_projected[:, 2] / 4
            [pp.remove() for pp in self.axes.get_figure().findobj(Path3DCollection) if pp.get_label() not in self.color_list]
            self.axes.scatter(x, y, z, marker='o', s=10, c=curr_label, alpha=0.8, label=i)
            color = self.color_list[int(self.labels[i])-1]
            center = centers[int(self.labels[i]-1)]
            for ll in self.axes.get_figure().findobj(Line2D):
                try:
                    if ll.get_label() != color: ll.set_alpha(ll.get_alpha()*0.7)
                except TypeError:
                    pass
            for cl in classes:
                try:
                    if cl.get_label() != color: cl.set_alpha(cl.get_alpha()*0.2)
                    elif cl.get_label() == color: cl.set_alpha(1.0) 
                except TypeError:
                    pass
            if all(self.last_center != center):
                line = self.axes.plot([self.last_pts[:, 0], curr_projected[:, 0]], 
                                [self.last_pts[:, 1], curr_projected[:, 1]], 
                                [self.last_pts[:, 2], curr_projected[:, 2]],
                                color=color, lw=1.0, alpha=1.0, label=color)
            self.last_center = center
            self.last_pts = curr_projected
            self.fig.canvas.draw()
            self.fig.canvas.blit(self.axes.bbox)
            filename = '__frame%03d.png' % int(i)
            filenames.append(filename)
            self.fig.savefig(filename, dpi=100)
        subprocess.call('ffmpeg -framerate 25 -i __frame%03d.png -c:v libx264 ' + self.out_movie, shell=True)
        for fi in filenames:
            os.remove(fi)

    # def play(self, event):
    #     self.fig.canvas.draw

    def update(self, i):
        def update_3d_arrows(color, i, ty):
            i = int(i)
            if ty == Line2D:
                for o in self.axes.get_figure().findobj(ty):
                    try:
                        if int(o.get_label()) != i and self.last_color != color:
                            current_alpha = o.get_alpha()
                            o.set_alpha(0.75*current_alpha)
                        elif int(o.get_label()) == i:
                            o.set_alpha(1.0)
                    except ValueError:
                        pass
            elif ty == Path3DCollection:
                for o in self.axes.get_figure().findobj(ty):
                    try:
                        if int(o.get_label()) in range(i-10, i+1):
                            o.set_alpha(1.0)
                        elif int(o.get_label()) not in range(i-10, i+1):
                            o.set_alpha(0.0)
                    except ValueError:
                        pass
        self.axes.view_init(elev=30., azim=i)
        color = self.color_list[int(self.labels[i])-1]
        update_3d_arrows(color, i, Line2D)
        update_3d_arrows(color, i, Path3DCollection)
        self.frame_no.set_text('Frame #: %d' % int(i))
        self.last_color = color
        return [arr for arr in self.axes.get_figure().findobj(Line2D)] + [self.frame_no] +\
        [arr for arr in self.axes.get_figure().findobj(Path3DCollection)]

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