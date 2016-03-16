from extimports import *

class Analyze(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.t = 0
        self.dpi = 200
        self.fig = Figure((5.0, 3.0), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.data_arr = dict()
        self.cond_arr = dict()
        self.neurons = list()
        self.conditions = list()
        self.__do_layout()

    def to_freq(self, data):
        nr_pts = 1e3
        vals = np.fromiter(itertools.chain.from_iterable(data.values()),dtype=np.float)
        time_space = np.linspace(min(vals), max(vals), nr_pts)
        delta = time_space[1] - time_space[0]
        np.insert(time_space, 0, time_space[0] - delta)
        np.insert(time_space, -1, time_space[-1] + delta)
        freq = np.zeros((max(data.keys())+1, nr_pts))
        for neuron, datum in data.items():
            for ii in np.arange(nr_pts):
                sum1 = np.sum(datum[datum > time_space[ii - 1]])
                sum2 = np.sum(datum[datum < time_space[ii]])
                freq[neuron, ii] = np.divide(sum1 + sum2, delta)
        freq = np.divide(freq - np.tile(np.mean(freq), (1, len(freq.T))), 
                         np.tile(np.std(freq), (1, len(freq.T))))
        freq = (1.0 + np.tanh(freq)) / 2.0
        return freq

    def load_data(self, filenames):
        data = dict()
        for ii, filename in enumerate(filenames):
            if ii == 0:
                tagname = filename.split('_')[0]
            if filenames[ii-1].split('_')[0] == filenames[ii].split('_')[0]:
                datum = np.loadtxt(filename,skiprows=2)
                data.update({ii: datum})
        freq = self.to_freq(data)
        pca = PCA(n_components=3)
        pca.fit(freq.T)
        freq = pca.transform(freq.T)
        freq = normalize(freq)
        self.data_arr.update({tagname: freq})
        np.savetxt(tagname + "_projected_freq.txt", freq)
        if self.t == 0:
            self.init_plot()
            self.t += 1

    def load_conditions(self, filenames):
        for filename in filenames:
            conds = np.loadtxt(filename)
            self.cond_arr.update({filename: conds})

    def plot_selected(self, event):
        ## TODO: define selected_dat, selected_labels
        labelled_data = self.class_creation(selected_labels, selected_dat)
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        for class_label in labelled_data.keys():
            current_class = labelled_data[class_label]
            pca = PCA(n_components=3)
            pca.fit(current_class)
            projected_class = normalize(pca.transform(current_class))
            projected_class = current_class*projected_class
            x = projected_class[:, 0]
            y = projected_class[:, 1]
            z = projected_class[:, 2]

            self.axes.scatter(x, y, z, c=color_list[int(class_label)-1], 
                marker='.', edgecolor='k')
            center, radii, rotation = EllipsoidTool().getMinVolEllipse(projected_class)
            """ 
            EllipsoidTool from:
            https://github.com/minillinim/ellipsoid 
            """
            EllipsoidTool().plotEllipsoid(center, radii, rotation, ax=self.axes, plotAxes=True, 
                                        cageColor=color_list[int(class_label)-1], cageAlpha=0.5)
        self.canvas.draw()

    def init_plot(self):
        self.axes = self.fig.add_subplot(111, projection='3d')
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
        except IndexError:
            init_labels = np.loadtxt('pdat_labels.txt')
        labelled_data = self.class_creation(init_labels, init_dat)
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        for class_label in labelled_data.keys():

            current_class = labelled_data[class_label]
            pca = PCA(n_components=3)
            pca.fit(current_class)
            projected_class = normalize(pca.transform(current_class))
            projected_class = current_class*projected_class
            x = projected_class[:, 0]
            y = projected_class[:, 1]
            z = projected_class[:, 2]

            self.axes.scatter(x, y, z, c=color_list[int(class_label)-1], 
                marker='.', edgecolor='k')
            center, radii, rotation = EllipsoidTool().getMinVolEllipse(projected_class)
            """ 
            EllipsoidTool from:
            https://github.com/minillinim/ellipsoid 
            """
            EllipsoidTool().plotEllipsoid(center, radii, rotation, ax=self.axes, plotAxes=True, 
                                        cageColor=color_list[int(class_label)-1], cageAlpha=0.5)
        self.canvas.draw()

    def pca_selected(self, event):
        pca = PCA(n_components=3)
        pca.fit(current_class)
        projected_class = normalize(pca.transform(current_class))
        projected_class = current_class*projected_class
        x = projected_class[:, 0]
        y = projected_class[:, 1]
        z = projected_class[:, 2]
        pass

    def mda_selected(self, event):
        pass

    def kmeans_selected(self, event):
        pass

    def class_creation(self, labels, data):
        classes = dict()
        for label in range(int(min(labels)), int(max(labels))+1):
            classes[label] = data[labels==label,:]
        return classes

    def __do_layout(self):
        sizer_1 = wx.BoxSizer()
        sizer_1.Add(self.canvas)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()