from extimports import *
from mda import MDA

class Analyze(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.t = 0
        self.dpi = 150

        self.data = None
        self.labels = None
        self.waveform_names = None
        self.projected_data = None
        self.data_dir = ''
        self.export_dir = ''
        self.prefix = ''
        
        self.fig = Figure((5.5, 3.5), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        
        self.create_listbox()

        self.save_title = wx.StaticText(self, -1, "Save as...", (80, 10))
        self.save_button = wx.Button(self, -1, "Export Image and Projection")
        self.save_button.Bind(wx.EVT_BUTTON, self.save_as)

        self.__do_layout()

    def save_as(self, evt):
        alg = self.alg_choice.GetString(self.alg_choice.GetSelection())
        output_path = os.path.join(self.export_dir, alg+'_'+'.png')
        self.fig.savefig(rename_out(output_path), dpi='figure')
        output_path2 = os.path.join(self.export_dir, alg+"_projected"+'_.txt')
        np.savetxt(rename_out(output_path2), self.projected_data)

    def create_listbox(self):

        self.alg_title = wx.StaticText(self, -1, "Analyze with...:", (80, 10))
        self.alg_choice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, ["PCA", "MDA", "ICA", 
                                                                         "k-Means (PCA)", "k-Means (ICA)", 
                                                                         "k-Means (MDA)", "GMM (PCA)"])
        self.alg_choice.SetSelection(0)
        self.Bind(wx.EVT_CHOICE, self.plot_selected, self.alg_choice) # Algorithm selection


    def plot_selected(self, event):
        self.axes.cla()
        self.axes.set_axis_bgcolor('white')
        self.axes.set_title('PCA: {}'.format(self.prefix), size=10)

        self.axes.set_xlabel('PC1',size=5)
        self.axes.set_ylabel('PC2',size=5)
        self.axes.set_zlabel('PC3',size=5)
        
        plt.setp(self.axes.get_xticklabels(), fontsize=4)
        plt.setp(self.axes.get_yticklabels(), fontsize=4)
        plt.setp(self.axes.get_zticklabels(), fontsize=4)
        
        selected_alg = self.alg_choice.GetString(self.alg_choice.GetSelection())
        selected_dat = self.data
        selected_labels = self.labels

        if selected_dat.shape[0] < selected_dat.shape[1]: selected_dat = selected_dat.T

        if selected_alg in ['PCA', 'MDA']:
            if selected_alg == 'PCA':
                self.pca_selected(selected_dat, selected_labels)
            elif selected_alg == 'MDA':
                self.mda_selected(selected_dat, selected_labels)
        elif selected_alg == 'k-Means (PCA)':
            self.kmeans_selected(selected_dat, labels=selected_labels, alg='PCA')
        elif selected_alg == 'k-Means (ICA)':
            self.kmeans_selected(selected_dat, labels=selected_labels, alg='ICA')
        elif selected_alg == 'k-Means (MDA)':
            self.kmeans_selected(selected_dat, labels=selected_labels, alg='MDA')
        elif selected_alg == 'GMM (PCA)':
            self.gmm_selected(selected_dat, labels=selected_labels)
        elif selected_alg == 'ICA':
            self.ica_selected(selected_dat, selected_labels)

    def init_plot(self):
        self.axes = self.fig.add_axes((0, 0, 1, 1), projection='3d')
        self.axes.set_axis_bgcolor('white')
        self.axes.set_title('PCA: {}'.format(self.prefix), size=10)
        self.axes.set_xlabel('PC1',size=5)
        self.axes.set_ylabel('PC2',size=5)
        self.axes.set_zlabel('PC3',size=5)
        plt.setp(self.axes.get_xticklabels(), fontsize=4)
        plt.setp(self.axes.get_yticklabels(), fontsize=4)
        plt.setp(self.axes.get_zticklabels(), fontsize=4)

        init_dat = self.data
        init_labels = self.labels

        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        self.pca_selected(init_dat, init_labels, toplot=True)

    def pca_selected(self, data, labels, toplot=True):
        self.axes.set_title('PCA: {}'.format(self.prefix), size=10)
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        if data.shape[0] < data.shape[1]: data = data.T
        pca = PCA(n_components=3)
        projected = pca.fit_transform(data)
        classes = list()
        for class_label in set(labels):
            projected_class = projected[labels==class_label,:]
            x = projected_class[:, 0]
            y = projected_class[:, 1]
            z = projected_class[:, 2]
            if toplot:
                curr_ = self.axes.scatter(x, y, z, c=color_list[int(class_label)-1], 
                                  marker='o', edgecolor='k', label=unicode(int(class_label)))
                classes.append(curr_)
        self.axes.legend(handles=classes, loc=3,
         scatterpoints=1, ncol=len(set(labels)), fontsize=4.5, 
         labels=self.waveform_names, frameon=True
         )
        self.projected_data = projected
        self.canvas.draw()

    def ica_selected(self, data, labels, toplot=True):
        self.axes.set_title('ICA: {}'.format(self.prefix), size=10)
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        if data.shape[0] < data.shape[1]: data = data.T
        ica = FastICA(n_components=3, max_iter=1000)
        projected = ica.fit_transform(data)
        classes = list()
        for class_label in set(labels):
            projected_class = projected[labels==class_label,:]
            x = projected_class[:, 0]
            y = projected_class[:, 1]
            z = projected_class[:, 2]
            if toplot:
                curr_=self.axes.scatter(x, y, z, c=color_list[int(class_label)-1], 
                                  marker='o', edgecolor='k', label=unicode(int(class_label)))
                classes.append(curr_)
        self.axes.legend(handles=classes, loc=3,
         scatterpoints=1, ncol=len(set(labels)), fontsize=4.5, 
         labels=self.waveform_names, frameon=True
         )
        self.projected_data = projected
        self.canvas.draw()

    def mda_selected(self, data, labels):
        self.axes.set_title('MDA: {}'.format(self.prefix), size=10)
        mda = MDA(data, labels)
        moutput = mda.fit_transform(test_percent=30.0)
        mda_labels, transformed_data = moutput
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        self.axes.set_xlabel('D1',size=5)
        self.axes.set_ylabel('D2',size=5)
        self.axes.set_zlabel('D3',size=5)
        classes = list()
        for ii in set(labels):
            d = transformed_data[mda_labels==ii, 0:3]
            curr_=self.axes.scatter(d[:, 0], d[:, 1], d[:, 2], c=color_list[int(ii-1)], 
                              marker='o', edgecolor='k', label=unicode(int(ii)))
            classes.append(curr_)
        self.axes.legend(handles=classes, loc=3,
         scatterpoints=1, ncol=len(set(labels)), fontsize=4.5, 
         labels=self.waveform_names, frameon=True
         )
        self.projected_data = transformed_data[:, 0:3]
        self.canvas.draw()

    def kmeans_selected(self, selected_data, labels=None, alg='PCA'):
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']

        def davies_bouldin_index(A, B):
            # https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
            p = 2

            if A.size > B.size: B = bezier(B, res=A.shape[0], dim=B.shape[1])
            elif A.size < B.size: A = bezier(A, res=B.shape[0], dim=A.shape[1])
            
            T_A = A.shape[0]
            A_c = np.mean(A, 0)
            S_A = (1.0 / T_A) * np.linalg.norm(A_c - A, p)

            T_B = B.shape[0]
            B_c = np.mean(B, 0)
            S_B = (1.0 / T_B) *  np.linalg.norm(B_c - B, p)
            
            M_AB = np.linalg.norm(A_c - B_c, p)
            
            with np.errstate(divide='ignore'):
                R_AB = float(S_A + S_B) / M_AB
                if np.isinf(R_AB): R_AB = 1.0
            return R_AB

        X = selected_data
        if alg == 'PCA':
            pca = PCA(n_components=3)
            projected = pca.fit_transform(X)
        elif alg == 'ICA':
            pca = FastICA(n_components=3)
            projected = pca.fit_transform(X)
        elif alg == 'MDA':
            mda = MDA(X, labels)
            test_labels, y_test = mda.fit_transform()
            projected = y_test[:, 0:3]
            labels = test_labels

        starts = list()
        for lll in set(labels):
            starts.append(np.mean(projected[labels==lll, :], 0))
        
        km = KMeans(n_clusters=len(set(labels)), init=np.asarray(starts), n_init=1)
        y_pred = km.fit_predict(projected, labels)

        modlab = labels - 1.0
        complist = list()
        for alab in set(modlab):
            for blab in set(y_pred):
                A = projected[modlab==alab,:]
                B = projected[y_pred==blab,:]
                complist.append((alab, blab, davies_bouldin_index(A, B)))

        y_corr = y_pred.copy()
        for ll in set(y_pred):
            # Kmeans predicted label
            clab = [li for li in complist if li[1] == ll]
            # Closest match
            best_c = max(clab, key=itemgetter(2))
            # Set closest match as the new label
            y_corr[y_corr==ll] = best_c[0]

        colist = list()
        for ix, yl in enumerate(y_corr):
            # If the two match in label, use green
            if y_pred[ix] == yl: col = 'g'
            # otherwise, use red
            else:
                col = 'r'
            colist.append(col)
        self.axes.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                          c=colist, marker='o', s=30)

        self.axes.set_title('k-Means (%s): %s \n %01f %% correct' % (alg, self.prefix, 100.00*colist.count('g')/float(len(colist))), size=10)
        
        self.canvas.draw()

    def gmm_selected(self, selected_data, labels=None):
        self.axes.set_title('GMM: {}'.format(self.prefix), size=10)
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        X = selected_data
        pca = PCA(n_components=3)
        projected = pca.fit_transform(X)

        gmm = GMM(n_components=len(set(labels)), random_state=0, covariance_type='diag')
        y_pred = gmm.fit_predict(projected, labels)
        # print gmm.score_samples(projected)[0]
        # GMM plot
        self.axes.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                              c=y_pred, marker='o', s=30)
    
        self.canvas.draw()


    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)

        sizer_1.Add(self.alg_title, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.alg_choice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)

        sizer_1.Add(self.save_title, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.save_button, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()