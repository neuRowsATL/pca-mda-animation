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
        self.in_args = tuple()
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
        self.alg_choice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, ["PCA", "MDA", "ICA", "k-Means (PCA)", "GMM (PCA)"])
        self.alg_choice.SetSelection(0)
        self.Bind(wx.EVT_CHOICE, self.plot_selected, self.alg_choice) # Algorithm selection

    def to_freq(self, data):
        nr_pts = 1e3

        # vals = np.fromiter(itertools.chain.from_iterable(data.values()),dtype=np.float32)
        # for neuron, values in data.items():
        #     std_thresh = np.std(values) * 2
        #     mean_val = np.mean(values)
        #     data[neuron] = values[np.where((values <= mean_val + std_thresh) & (values >= mean_val - std_thresh))]
        vals = np.fromiter(itertools.chain.from_iterable(data.values()),dtype=np.float32)
        if len(vals) > 0:
            time_space = np.linspace(min(vals), max(vals), nr_pts, endpoint=True)
            delta = time_space[1] - time_space[0]
            time_space = np.insert(time_space, 0, time_space[0] - delta)
            time_space = np.insert(time_space, -1, time_space[-1] + delta)
            freq = np.zeros((int(max(data.keys())+1), int(nr_pts)))
            for neuron, datum in data.items():
                for ii in np.arange(nr_pts):
                    ii = int(ii)
                    count = len(datum[np.where((datum < time_space[ii + 1]) & (datum > time_space[ii]))])
                    freq[neuron, ii] = np.divide(count, delta)
            fmean = np.mean(freq, 1)
            fstd = np.std(freq, 1)
            freq = np.array((freq - np.expand_dims(fmean, axis=1)) /
                   np.expand_dims(fstd,axis=1))
            freq = (1.000 + np.tanh(freq)) / 2.000
            freq = freq.T

            # np.random.seed(0)
            # train_ix = np.random.random_integers(0, len(freq)-1, int(0.4*len(freq)))
            # test_ix = np.in1d(np.asarray(list(range(0, len(freq)))), train_ix, invert=True)
            # X_train = freq[train_ix,:]
            # X_test = freq[test_ix,:]
            # svm = OneClassSVM(random_state=0, kernel='rbf', nu=0.5)
            # svm.fit(X_train)
            # y_pred_train = svm.predict(X_train)
            # y_pred_test = svm.predict(X_test)
            # # y_pred = svm.predict(freq)

            # n_error_train = y_pred_train[y_pred_train == -1].shape[0] / float(X_train.shape[0])
            # n_error_test = y_pred_test[y_pred_test == -1].shape[0] / float(X_test.shape[0])
            # print(n_error_train, n_error_test)

            # self.in_args = np.where(y_pred_test == 1)
            # print(len(y_pred_test == 1))
            # return X_test[self.in_args]
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
            save_name = tagname + "_normalized_freq.txt"
            np.savetxt(save_name, freq)
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
                    return init_arr
        elif t == 'Cond':
            for ii, _ in enumerate(self.cond_arr.keys()):
                if self.cond_arr.keys()[ii].split('\\')[-1] == keyname:
                    init_arr = self.cond_arr[_]
                    return init_arr[self.in_args]

    def get_selection(self, box, t):
        selected = [box.GetString(box.GetSelection())]
        selarr = dict()
        # if len(selected) == 1:
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
        # selected_labels = selected_labels[self.in_args]
        if selected_dat.shape[0] < selected_dat.shape[1]: selected_dat = selected_dat.T
        if len(self.cond_arr.keys()) < 1 and selected_alg in ['PCA', 'MDA']:
            print("To use MDA or PCA, please select both frequency and labels.")
            return
        if len(self.cond_arr.keys()) > 0 and len(self.data_arr.keys()) > 0 and selected_alg in ['PCA', 'MDA']:
            if selected_alg == 'PCA':
                self.pca_selected(selected_dat, selected_labels)
            elif selected_alg == 'MDA':
                self.mda_selected(selected_dat, selected_labels)
        elif len(self.cond_arr.keys()) > 0 and len(self.data_arr.keys()) > 0 and selected_alg == 'k-Means (PCA)':
            self.kmeans_selected(selected_dat, labels=selected_labels)
        elif selected_alg == 'GMM (PCA)':
            self.gmm_selected(selected_dat, labels=selected_labels)
        elif selected_alg == 'ICA':
            self.ica_selected(selected_dat, selected_labels)

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
            init_labels = self.cond_arr[self.cond_arr.keys()[0]][self.in_args]
            # labelled_data = self.class_creation(init_labels, init_dat)
            color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
            self.pca_selected(init_dat, init_labels, toplot=True)
            self.legend = self.axes.legend(frameon=True, loc='upper left', scatterpoints=1, ncol=2, fontsize=8, bbox_to_anchor=(0, 0))
        except IndexError:
            return

    def pca_selected(self, data, labels, toplot=True):
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        if data.shape[0] < data.shape[1]: data = data.T
        pca = PCA(n_components=3)
        projected = pca.fit_transform(data)
        for class_label in set(labels):
            projected_class = projected[labels==class_label,:]
            x = projected_class[:, 0]
            y = projected_class[:, 1]
            z = projected_class[:, 2]
            if toplot:
                self.axes.scatter(x, y, z, c=color_list[int(class_label)-1], 
                                  marker='o', edgecolor='k', label=str(int(class_label)))
                # center, radii, rotation = EllipsoidTool().getMinVolEllipse(projected_class)
                # EllipsoidTool().plotEllipsoid(center, radii, rotation, ax=self.axes, plotAxes=False, 
                #                             cageColor=color_list[int(class_label)-1], cageAlpha=0.1)
        self.canvas.draw()

    def ica_selected(self, data, labels, toplot=True):
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        if data.shape[0] < data.shape[1]: data = data.T
        ica = FastICA(n_components=3, max_iter=1000)
        projected = ica.fit_transform(data)
        for class_label in set(labels):
            projected_class = projected[labels==class_label,:]
            x = projected_class[:, 0]
            y = projected_class[:, 1]
            z = projected_class[:, 2]
            if toplot:
                self.axes.scatter(x, y, z, c=color_list[int(class_label)-1], 
                                  marker='o', edgecolor='k', label=str(int(class_label)))
                # center, radii, rotation = EllipsoidTool().getMinVolEllipse(projected_class)
                # EllipsoidTool().plotEllipsoid(center, radii, rotation, ax=self.axes, plotAxes=False, 
                #                             cageColor=color_list[int(class_label)-1], cageAlpha=0.1)
        self.canvas.draw()

    def mda_selected(self, data, labels):
        mda = MDA(data, labels)
        train_labels, y_train, test_labels, y_test = mda.fit_transform()
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        self.axes.set_xlabel('D1',size=5)
        self.axes.set_ylabel('D2',size=5)
        self.axes.set_zlabel('D3',size=5)
        for ii in set(labels):
            test_val = y_test[test_labels==ii, 0:3]
            self.axes.scatter(test_val[:, 0], test_val[:, 1], test_val[:, 2], c=color_list[int(ii-1)], 
                      marker='o', edgecolor='k', label=str(ii))
            selected_projection = y_train[train_labels==ii, 0:3]
            # _, _, v = np.linalg.svd(selected_projection)
            # recentered = np.dot(selected_projection, np.linalg.inv(v))
            # v = v[0:3, 0:3]
            x = selected_projection[:, 0]
            y = selected_projection[:, 1]
            z = selected_projection[:, 2]
            self.axes.scatter(x, y, z, c=color_list[int(ii-1)], 
                      marker='o', edgecolor='k', label=str(ii))
            # center, radii, rotation = EllipsoidTool().getMinVolEllipse(selected_projection, 0.001)
            # EllipsoidTool().plotEllipsoid(center, radii, rotation, ax=self.axes, plotAxes=False, 
            #                     cageColor=color_list[int(ii)-1], cageAlpha=0.1)
        self.canvas.draw()

    def kmeans_selected(self, selected_data, labels=None):
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        def cosine_sim(A, B):
            # https://en.wikipedia.org/wiki/Cosine_similarity
            if A.size > B.size: B = bezier(B, res=A.shape[0], dim=B.shape[1])
            elif A.size < B.size: A = bezier(A, res=B.shape[0], dim=A.shape[1])
            return np.trace(np.dot(A.T, B)) / (np.linalg.norm(A)*np.linalg.norm(B))
        def chung_capps_index(A, B):
            if A.size > B.size: B = bezier(B, res=A.shape[0], dim=B.shape[1])
            elif A.size < B.size: A = bezier(A, res=B.shape[0], dim=A.shape[1])
            # compute cosine similarity
            cs = cosine_sim(A, B)
            with np.errstate(divide='ignore'):
                diff = np.linalg.norm(A - B)
                step2 =  np.sum(np.std(A) + np.std(B)) / diff
                if np.isinf(step2): step2 = 1.0
            return cs + step2

        def modified_cci(A, B):
            if A.size > B.size: B = bezier(B, res=A.shape[0], dim=B.shape[1])
            elif A.size < B.size: A = bezier(A, res=B.shape[0], dim=A.shape[1])
            with np.errstate(divide='ignore'):
                # diff = np.linalg.norm(A - B)
                # diff = np.sum([abs(g0 - f0) for g0, f0 in zip(A, B)])
                diff = np.sum(cdist(A, B, 'seuclidean'))
                # kurtA = (A - np.expand_dims(np.mean(A, 0), 0))**3 / np.std(A, 0)**(3)
                # kurtB = (B - np.expand_dims(np.mean(B, 0), 0))**3 / np.std(B, 0)**(3)
                step2 =  np.sum(np.std(A, 0) + np.std(B, 0)) / diff
                # step2 = np.sum(kurtA + kurtB) / diff
                # if np.isinf(step2): step2 = 1.0
            return step2

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

            # return (np.tanh(R_AB) + 1.0) / 2.0
            return R_AB

        def spca(A, B):
            # A. Sinhal, D. Seborg.
            # Matching patterns from historical data using pca and distance similarity factors,
            # 2001.
            
            pcaA = PCA(n_components=2)
            pcaB = PCA(n_components=2)
            
            A_p = pcaA.fit_transform(A)
            B_p = pcaB.fit_transform(B)
            
            return cosine_sim(A_p, B_p)

        X = selected_data
        pca = PCA(n_components=3)
        projected = pca.fit_transform(X)

        # range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
        # avgs = list()
        # for k in range_n_clusters:
        #     kmeans = KMeans(n_clusters=k, random_state=0)
        #     cluster_labels = kmeans.fit_predict(X)
        #     sil_avg = silhouette_score(X, cluster_labels)
        #     avgs.append((k, sil_avg))
        # best_k = max(avgs, key=itemgetter(1))
        
        # km = KMeans(n_clusters=best_k[0], random_state=0)
        
        # km = KMeans(n_clusters=len(set(labels)), random_state=0)

        starts = list()
        for lll in set(labels):
            starts.append(np.mean(projected[labels==lll, :], 0))
        km = KMeans(n_clusters=len(set(labels)), init=np.asarray(starts), n_init=1)
        
        y_pred = km.fit_predict(projected, labels)


        complist = list()
        for alab in set(labels):
            for blab in set(y_pred):
                A = projected[labels==alab,:]
                B = projected[y_pred==blab,:]
                complist.append((alab, blab, davies_bouldin_index(A, B)))
        y_corr = y_pred.copy()
        for ll in set(y_pred):
            # Kmeans predicted label
            clab = [li for li in complist if li[1] == ll]
            # Closest match
            best_c = max(clab, key=itemgetter(-1))
            # Set closest match as the new label
            y_corr[y_corr==ll] = best_c[0] - 1
            # print best_c[0]
        colist = list()
        for ix, yl in enumerate(y_corr):
            # If the two match in label, use green
            if y_pred[ix] == yl: col = 'g'
            # otherwise, use red
            else: 
                col = 'r'
                # print yl
            colist.append(col)
        self.axes.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                          c=colist, marker='o', s=30)
        # print colist.count('g')/float(len(colist))
        
        self.canvas.draw()

    def gmm_selected(self, selected_data, labels=None):
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        X = selected_data
        pca = PCA(n_components=3)
        projected = pca.fit_transform(X)

        gmm = GMM(n_components=7, random_state=0, covariance_type='diag')
        y_pred = gmm.fit_predict(projected, labels)
        # GMM plot
        self.axes.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                              c=y_pred, marker='o', s=30)
    
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