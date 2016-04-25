from extimports import *

class Compareize(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.fig = Figure((5.5, 3.5), dpi=150)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.labels = list()
        self.algList = ['jaccard', 'dbi', 'spca', 'cos']
        self.alg = 'jaccard'
        self.min_class = 5
        self.algtitle = wx.StaticText(self, -1, "Choose Similarity Metric:", (80, 10))
        self.algchoice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, self.algList)
        self.Bind(wx.EVT_CHOICE, self.plotting, self.algchoice)
        self.algchoice.SetSelection(0)
        self.save_button = wx.Button(self, -1, "Save Image as PNG", size=(800, 100))
        self.Bind(wx.EVT_BUTTON, self.save_fig, self.save_button)
        self.__do_layout()

    def get_data(self):
        fname = '_normalized_freq.txt'
        dname = [f for f in os.listdir('./Data/') if fname in f]
        if len(dname) > 0:
            data = np.loadtxt('./Data/'+dname[0])
            if data.shape[0] < data.shape[1]: data = data.T
            return data
        return None

    def waveforms(self):
        waveform_names = {
                          5: 'inf_sine',
                          2: 'CL',
                          3: 'low_sine',
                          1: 'no_sim',
                          4: 'top_sine',
                          6: 'tugs_ol',
                          7: 'other'}
        return list(waveform_names.values())

    def jaccard_index(self, A, B):
        """ http://stackoverflow.com/questions/20230384/find-indexes-of-matching-rows-in-two-2-d-arrays
        """
        # https://en.wikipedia.org/wiki/Jaccard_index
        A_B = len(np.array(np.all((A[:,None,:]==B[None,:,:]),axis=-1).nonzero()).T.tolist())
        jsc = float(A_B) / float(A.shape[0] + B.shape[0] - A_B)
        return jsc

    def davies_bouldin_index(self, A, B):
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

        M_AB = np.linalg.norm(A - B, p)
        if M_AB == 0.0: M_AB = 1e-6
        R_AB = float(S_A + S_B) / M_AB

        return (np.tanh(R_AB) + 1.0) / 2.0

    def spca(self, A, B):
        # A. Sinhal, D. Seborg.
        # Matching patterns from historical data using pca and distance similarity factors,
        # 2001.
        
        min_class = self.min_class
        pcaA = PCA(n_components=min_class)
        pcaB = PCA(n_components=min_class)
        
        A_p = pcaA.fit_transform(A)
        B_p = pcaB.fit_transform(B)
        
        return (np.tanh(self.cosine_sim(A_p, B_p)) + 1.0) / 2.0

    def cosine_sim(self, A, B):
        # https://en.wikipedia.org/wiki/Cosine_similarity
        if A.size > B.size: B = bezier(B, res=A.shape[0], dim=B.shape[1])
        elif A.size < B.size: A = bezier(A, res=B.shape[0], dim=A.shape[1])
        return np.trace(np.dot(A.T, B)) / (np.linalg.norm(A)*np.linalg.norm(B))

    def compare(self):
        comp_algs = {
                'jaccard': self.jaccard_index,
                'spca': self.spca,
                'cos': self.cosine_sim,
                'dbi': self.davies_bouldin_index
                }
        labels = np.loadtxt(self.labels[0])
        data = self.get_data()
        if data is not None:
            self.min_class = min([list(labels).count(i) for i in range(1, 1+len(set(labels)))])
            comps = list()
            do_comp = comp_algs[self.alg]
            for li in set(labels):
                cl = list()
                for li2 in set(labels):
                    A = data[labels==li, :]
                    B = data[labels==li2, :]
                    cl.append(do_comp(A, B))
                comps.append(cl)
            return np.asarray(comps)
        return None

    def plotting(self, event=None):
        self.alg = self.algList[self.algchoice.GetSelection()]
        comparison = self.compare()
        if comparison is not None:
            titles = {
                'jaccard': 'Jaccard Index',
                'spca': 'PCA Cosine Similarity',
                'cos': 'Cosine Similarity',
                'dbi': 'Davies Bouldin Index'
            }
            labels = np.loadtxt(self.labels[0])
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.set_title('Metric:\n' + titles[self.alg], size=7)
            p = ax.pcolormesh(comparison)
            ax.set_xticks(np.arange(comparison.shape[1])+0.5, minor=False)
            ax.set_yticks(np.arange(comparison.shape[0])+0.5, minor=False)
            ax.set_xticklabels(self.waveforms(), minor=False)
            ax.set_yticklabels(self.waveforms(), minor=False)

            plt.setp(ax.get_xticklabels(), fontsize=5)
            plt.setp(ax.get_yticklabels(), fontsize=5)

            ax.set_xlabel('Class 1', size=6)
            ax.set_ylabel('Class 2', size=6)
            cbar = self.fig.colorbar(p)
            cbar.ax.tick_params(labelsize=5)
            self.fig.tight_layout()
            self.canvas.draw()

    def save_fig(self, event):
        self.fig.savefig('Class_sim_'+self.alg+'.png', dpi=200)

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)

        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(5)
        sizer_1.Add(self.algtitle, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.algchoice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)

        sizer_1.AddSpacer(7)

        sizer_1.Add(self.save_button, 0, wx.ALIGN_CENTER, 10)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()