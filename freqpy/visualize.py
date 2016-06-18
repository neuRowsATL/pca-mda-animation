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
        
        self.data = None
        self.labels = None
        self.data_dir = ''
        self.export_dir = ''
        self.title_ = ''
        self.out_movie = ''

        self.ax_labels = list()
        self.vis_selected = False
        self.waveform = None

         # Algorithm selection
        self.alg_title = wx.StaticText(self, -1, "Choose Projection:", (80, 10))
        self.alg_choice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, ["PCA", "MDA", "ICA", "k-Means", "GMM"])
        self.alg_choice.SetSelection(0)
        self.Bind(wx.EVT_CHOICE, self.plot_selected, self.alg_choice)

        # Choose DPI
        self.dpi_title = wx.StaticText(self, -1, "Select Video Quality:", (80, 10))
        self.dpi_choice = wx.Choice(self, -1, (80, 30), wx.DefaultSize, ["Low (100 dpi)", "Medium (150 dpi)", "High (200 dpi)"])
        self.dpi_choice.SetSelection(0)
        self.Bind(wx.EVT_CHOICE, self.set_dpi, self.dpi_choice)

        self.save_button = wx.Button(self, -1, "Export as MPEG.", size=(800, 100))
        
        self.__do_layout()

    def init_viz(self):
        init_labels = self.labels
        self.pca_selected(labels=init_labels)

    def set_dpi(self, event):
        dpis = [100, 150, 200]
        self.dpi = dpis[self.dpi_choice.GetSelection()]

    def plot_selected(self, event=None):
        selected_alg = self.alg_choice.GetString(self.alg_choice.GetSelection())
        selected_dat = self.data
        selected_labels = self.labels

        if selected_dat.shape[0] < selected_dat.shape[1]: selected_dat = selected_dat.T

        if selected_alg in ['ICA', 'PCA', 'MDA']:
            if selected_alg == 'PCA':
                self.pca_selected(selected_dat, selected_labels)
            elif selected_alg == 'MDA':
                self.mda_selected(selected_dat, selected_labels)
            if selected_alg == 'ICA':
                self.ica_selected(selected_dat, selected_labels)
        elif selected_alg == 'k-Means':
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
        np.savetxt(os.path.join(self.data_dir, '_mda_labels.txt'), test_labels)
        np.savetxt('_mda_projected.txt', y_test[:, 0:3])

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
        np.savetxt(os.path.join(self.data_dir,'_kmeans_labels.txt'), y_pred)
        np.savetxt(os.path.join(self.data_dir,'_kmeans_projected.txt'), projected)

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
        np.savetxt(os.path.join(self.data_dir,'_gmm_labels.txt'), y_pred)
        np.savetxt(os.path.join(self.data_dir,'_gmm_projected.txt'), projected)

    def class_creation(self, labels, data):
        classes = dict()
        data = data.T
        for label in range(int(min(labels)), int(max(labels))+1):
            classes[label] = data[labels==label,:]
        return classes

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)

        sizer_1.Add(self.alg_title, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.alg_choice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)

        sizer_1.Add(self.dpi_title, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.dpi_choice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)

        sizer_1.AddSpacer(5)
        sizer_1.Add(self.save_button, 0, wx.ALIGN_CENTER, 10)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()