from extimports import *
from mda import MDA

class Visualize(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.dpi = 100
        self.color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        self.colors = {'k': (0.0, 0.0, 0.0), 'b': (0.0, 0.0, 1.0), 'm': (0.75, 0, 0.75), 
                       'r': (1.0, 0.0, 0.0), 'y': (0.75, 0.75, 0), 'w': (1.0, 1.0, 1.0), 
                       'g': (0.0, 0.5, 0.0), 'c': (0.0, 0.75, 0.75)}
        self.data_dir = ''
        self.vis_selected = False
        self.waveform = None
        self.data_arr = dict()
        self.cond_arr = dict()
        self.in_args = tuple()
        self.lb_arr = list()
        self.lb_condarr = list()
        self.neurons = list()
        self.conditions = list()
        self.files = list()
        self.create_choices()
        self.save_button = wx.Button(self, -1, "Export as MPEG.", size=(800, 100))
        self.__do_layout()

    def set_inargs(self, intup):
        self.in_args = intup

    def init_viz(self):
        try:
            init_dat = self.data_arr[self.data_arr.keys()[0]]
            init_labels = self.cond_arr[self.cond_arr.keys()[0]][self.in_args]
            # labelled_data = self.class_creation(init_labels, init_dat)
            self.pca_selected(labels=init_labels)
        except IndexError:
            pass

    def create_choices(self):
        sampleList = list()
        condList = list()
        for ii, k in enumerate(self.data_arr.keys()):
            prev_dat = self.data_arr.keys()[ii-1].split('\\')[-1].split('_')[0]
            curr_dat = self.data_arr.keys()[ii].split('\\')[-1].split('_')[0]
            if ii > 0 and prev_dat != curr_dat:
                sampleList.append(curr_dat)
            if ii == 0:
                sampleList.append(curr_dat)
        for ii, k in enumerate(self.cond_arr.keys()):
            prev_cond = self.cond_arr.keys()[ii-1].split('\\')[-1].split('_')[0]
            curr_cond = self.cond_arr.keys()[ii].split('\\')[-1].split('_')[0]
            if ii > 0 and prev_cond != curr_cond:
                condList.append(curr_cond)
            if ii == 0:
                condList.append(curr_cond)

        # Choose files
        self.lbtitle = wx.StaticText(self, -1, "Choose Frequency Data:", (80, 10))
        self.lb = wx.Choice(self, -1, (80, 50), wx.DefaultSize, sampleList)
        self.condtitle = wx.StaticText(self, -1, "Choose Condition File:", (80, 10))
        self.lb_cond = wx.Choice(self, -1, (80, 50), wx.DefaultSize, condList)

         # Algorithm selection
        self.alg_title = wx.StaticText(self, -1, "Analyze with...:", (80, 10))
        self.alg_choice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, ["PCA", "MDA", "ICA", "k-Means", "GMM"])
        self.alg_choice.SetSelection(0)
        self.Bind(wx.EVT_CHOICE, self.plot_selected, self.alg_choice)

        # Choose DPI
        self.dpi_title = wx.StaticText(self, -1, "Select Video Quality:", (80, 10))
        self.dpi_choice = wx.Choice(self, -1, (80, 30), wx.DefaultSize, ["Low (100 dpi)", "Medium (150 dpi)", "High (200 dpi)"])
        self.dpi_choice.SetSelection(0)
        self.Bind(wx.EVT_CHOICE, self.set_dpi, self.dpi_choice)

    def set_dpi(self, event):
        dpis = [100, 150, 200]
        self.dpi = dpis[self.dpi_choice.GetSelection()]

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
            for ii, k in enumerate(self.data_arr.keys()):
                prev_dat = self.data_arr.keys()[ii-1].split('_')[0]
                curr_dat = self.data_arr.keys()[ii].split('_')[0]
                if ii > 0 and prev_dat != curr_dat:
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
                    init_arr = self.cond_arr[_][self.in_args]
                    np.savetxt(self.data_dir+'inlier_labels.txt', init_arr)
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
        selected_alg = self.alg_choice.GetString(self.alg_choice.GetSelection())
        selected_dat = self.get_selection(self.lb, t='Data')
        selected_labels = self.get_selection(self.lb_cond, t='Cond')
        if selected_dat.shape[0] < selected_dat.shape[1]: selected_dat = selected_dat.T
        if len(self.cond_arr.keys()) < 1 and selected_alg in ['ICA', 'PCA', 'MDA']:
            print("Please select both frequency and labels.")
            return
        if len(self.cond_arr.keys()) > 0 and len(self.data_arr.keys()) > 0 and \
        selected_alg in ['ICA', 'PCA', 'MDA']:
            if selected_alg == 'PCA':
                self.pca_selected(selected_dat, selected_labels)
            elif selected_alg == 'MDA':
                self.mda_selected(selected_dat, selected_labels)
            if selected_alg == 'ICA':
                self.ica_selected(selected_dat, selected_labels)
        elif len(self.cond_arr.keys()) > 0 and len(self.data_arr.keys()) > 0 and \
         selected_alg == 'k-Means':
            self.kmeans_selected(selected_dat, labels=selected_labels)
        elif selected_alg == 'GMM':
            self.gmm_selected(selected_dat, labels=selected_labels)

    def pca_selected(self, labels, data=None):
        self.title_ = 'PCA'
        self.ax_labels = ['PC1', 'PC2', 'PC3']
        self.labels = labels
        self.out_movie = 'PCA_Anim.mp4'

    def ica_selected(self, data, labels):
        self.title_ = 'ICA'
        self.ax_labels = ['IC1', 'IC2', 'IC3']
        self.labels = labels
        self.out_movie = 'ICA_Anim.mpg'

    def mda_selected(self, data, labels):
        self.title_ = 'MDA'
        self.ax_labels = ['D1', 'D2', 'D3']
        self.labels = labels
        self.out_movie = 'MDA_Anim.mp4'
        mda = MDA(data, labels)
        train_labels, y_train, test_labels, y_test = mda.fit_transform()
        os.chdir(self.data_dir)
        np.savetxt('_mda_labels.txt', np.hstack((train_labels, test_labels)))
        np.savetxt('_mda_projected.txt', np.vstack((y_train[:, 0:3], y_test[:, 0:3])))
        os.chdir('..')

    def kmeans_selected(self, selected_data, labels=None):
        self.title_ = 'K-Means (PCA)'
        self.ax_labels = ['PC1', 'PC2', 'PC3']
        self.labels = labels
        self.out_movie = 'Kmeans_Anim.mp4'
        X = selected_data
        pca = PCA(n_components=3)
        projected = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=len(set(labels)), random_state=0)
        kmeans.fit(projected)
        y_pred = kmeans.labels_
        os.chdir(self.data_dir)
        np.savetxt('_kmeans_labels.txt', y_pred)
        np.savetxt('_kmeans_projected.txt', projected)
        os.chdir('..')

    def gmm_selected(self, selected_data, labels=None):
        self.title_ = 'GMM (PCA)'
        self.ax_labels = ['PC1', 'PC2', 'PC3']
        self.labels = labels
        self.out_movie = 'GMM_Anim.mp4'
        X = selected_data
        pca = PCA(n_components=3)
        projected = pca.fit_transform(X)
        gmm = GMM(n_components=len(set(labels)), random_state=0)
        y_pred = gmm.fit_predict(projected)
        os.chdir(self.data_dir)
        np.savetxt('_gmm_labels.txt', y_pred)
        np.savetxt('_gmm_projected.txt', projected)
        os.chdir('..')

    def class_creation(self, labels, data):
        classes = dict()
        data = data.T
        for label in range(int(min(labels)), int(max(labels))+1):
            classes[label] = data[labels==label,:]
        return classes

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.lbtitle, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.lb, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        sizer_1.Add(self.condtitle, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.lb_cond, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        sizer_1.Add(self.alg_title, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.alg_choice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        sizer_1.Add(self.dpi_title, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.dpi_choice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        sizer_1.AddSpacer(5)
        sizer_1.Add(self.save_button, 0, wx.ALIGN_CENTER, 10)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()