from extimports import *
from mda import MDA

class Analyze(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.t = 0
        self.dpi = 150
        self.fig = Figure((5.5, 3.5), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.data_arr = dict()
        self.cond_arr = dict()
        self.lb_arr = list()
        self.lb_condarr = list()
        self.neurons = list()
        self.conditions = list()
        self.create_listbox()
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
        vals = np.fromiter(itertools.chain.from_iterable(data.values()),dtype=np.float32)
        if len(vals) > 0:
            time_space = np.linspace(min(vals), max(vals), nr_pts, endpoint=True)
            delta = time_space[1] - time_space[0]
            time_space = np.insert(time_space, 0, time_space[0] - delta)
            time_space = np.insert(time_space, -1, time_space[-1] + delta)
            freq = np.zeros((max(data.keys())+1, nr_pts))
            for neuron, datum in data.items():
                for ii in np.arange(nr_pts):
                    count = len(datum[np.where((datum < time_space[ii  + 1]) & (datum > time_space[ii]))])
                    freq[neuron, ii] = np.divide(count, delta)
            mean_tile = np.tile(np.mean(freq, 1), (freq.shape[1], 1)).T
            freq1 = np.subtract(freq, mean_tile)
            std_tile = np.tile(np.std(freq, 1), (freq.shape[1], 1)).T
            freq = np.divide(freq1, std_tile)
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
        labelled_data = self.class_creation(selected_labels, selected_dat)
        if len(self.cond_arr.keys()) < 1 and selected_alg in ['PCA', 'MDA']:
            print("To use MDA or PCA, please select both frequency and labels.")
            return
        if len(self.cond_arr.keys()) > 0 and len(self.data_arr.keys()) > 0 and selected_alg in ['PCA', 'MDA']:
            if selected_alg == 'PCA':
                self.pca_selected(labelled_data)
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
        init_dat = self.data_arr[self.data_arr.keys()[0]]
        try:
            init_labels = self.cond_arr[self.cond_arr.keys()[0]]
            labelled_data = self.class_creation(init_labels, init_dat)
            color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
            self.pca_selected(labelled_data, toplot=True)
            self.legend = self.axes.legend(frameon=True, loc='upper left', scatterpoints=1, ncol=2, fontsize=8, bbox_to_anchor=(0, 0))
        except IndexError:
            return

    def pca_selected(self, labelled_data, toplot=True):
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        """ TODO: Correct... the PCA needs to be done before splitting classes.
                  That's why it looks so different from kMeans.
        """
        for class_label in labelled_data.keys():
            current_class = labelled_data[class_label]
            pca = PCA(n_components=3)
            projected_class = pca.fit_transform(current_class)
            x = projected_class[:, 0]
            y = projected_class[:, 1]
            z = projected_class[:, 2]
            if toplot:
                self.axes.scatter(x, y, z, c=color_list[int(class_label)-1], 
                                  marker='o', edgecolor='k', label=str(int(class_label)))
                center, radii, rotation = EllipsoidTool().getMinVolEllipse(projected_class)
                EllipsoidTool().plotEllipsoid(center, radii, rotation, ax=self.axes, plotAxes=False, 
                                            cageColor=color_list[int(class_label)-1], cageAlpha=0.7)
        self.canvas.draw()

    def mda_selected(self, data, labels):
        mda = MDA(data, labels)
        train_labels, y_train, test_labels, y_test = mda.fit_transform()
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
            EllipsoidTool().plotEllipsoid(center, radii, rotation, ax=self.axes, plotAxes=False, 
                                cageColor=color_list[int(ii)-1], cageAlpha=0.7)
        self.axes.autoscale_view(True, True, True)
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
        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)
        sizer_1.Add(self.lbtitle, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.lb, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        sizer_1.Add(self.condtitle, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.lb_cond, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        sizer_1.Add(self.alg_title, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.alg_choice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()