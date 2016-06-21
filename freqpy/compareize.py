from extimports import *
from mda import MDA

class Compareize(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.fig = Figure((5.5, 3.5), dpi=150)
        self.canvas = FigCanvas(self, -1, self.fig)
        
        self.labels = None
        self.data = None
        self.waveform_names = None

        self.export_dir = ''
        self.data_dir = ''

        self.algList = ['DBI', 'PCA Distance', 'Cosine Similarity']
        self.alg = 'DBI'

        self.axesList = ['Original/Original', 'K-Means/K-Means', 'Original/K-Means']
        self.axesCurr = 'Original/Original'

        self.proList = ['None', 'PCA', 'MDA', 'ICA']
        self.proCurr = 'None'

        self.class_title = wx.StaticText(self, -1, "Choose Axis1/Axis2:", (80, 10))
        self.class_choice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, self.axesList)
        self.Bind(wx.EVT_CHOICE, self.plotting, self.class_choice)
        self.class_choice.SetSelection(0)

        self.algtitle = wx.StaticText(self, -1, "Choose Similarity Metric:", (80, 10))
        self.algchoice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, self.algList)
        self.Bind(wx.EVT_CHOICE, self.plotting, self.algchoice)
        self.algchoice.SetSelection(0)

        self.protitle = wx.StaticText(self, -1, "Choose Projection:", (80, 10))
        self.prochoice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, self.proList)
        self.Bind(wx.EVT_CHOICE, self.plotting, self.prochoice)
        self.prochoice.SetSelection(0)

        self.save_button = wx.Button(self, -1, "Save as PNG", size=(100, 50))
        self.Bind(wx.EVT_BUTTON, self.save_fig, self.save_button)

        self.__do_layout()

    def get_data(self):
        return self.data

    def waveforms(self):
        return self.waveform_names

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

        M_AB = np.linalg.norm(A_c - B_c, p)

        with np.errstate(divide='ignore'):
            R_AB = float(S_A + S_B) / M_AB
            if np.isinf(R_AB): R_AB = 1.0

        # return (np.tanh(R_AB) + 1.0) / 2.0
        return R_AB

    def chung_capps_index(self, A, B):
        if A.size > B.size: B = bezier(B, res=A.shape[0], dim=B.shape[1])
        elif A.size < B.size: A = bezier(A, res=B.shape[0], dim=A.shape[1])
        
        # compute cosine similarity
        cs = self.cosine_sim(A, B)

        with np.errstate(divide='ignore'):
            diff = np.linalg.norm(A - B)
            step2 =  np.sum(np.std(A) + np.std(B)) / diff
            if np.isinf(step2): step2 = 1.0
        return cs * step2

    def spca(self, A, B):
        # A. Sinhal, D. Seborg.
        # Matching patterns from historical data using pca and distance similarity factors,
        # 2001.
        
        pcaA = PCA(n_components=3)
        pcaB = PCA(n_components=3)
        
        A_p = pcaA.fit_transform(A)
        B_p = pcaB.fit_transform(B)
        
        # return (np.tanh(self.cosine_sim(A_p, B_p)) + 1.0) / 2.0
        return self.cosine_sim(A_p, B_p)

    def cosine_sim(self, A, B):
        # https://en.wikipedia.org/wiki/Cosine_similarity
        if A.size > B.size: B = bezier(B, res=A.shape[0], dim=B.shape[1])
        elif A.size < B.size: A = bezier(A, res=B.shape[0], dim=A.shape[1])
        return np.trace(np.dot(A.T, B)) / (np.linalg.norm(A)*np.linalg.norm(B))

    def pca(self, data, labels):
        pca = PCA(n_components=3)
        proj = pca.fit_transform(data)
        return proj

    def mda(self, data, labels):
        mda = MDA(data, labels)
        moutput = mda.fit_transform()
        return moutput[1][:, 0:3]

    def ica(self, data, labels):
        ica = FastICA(n_components=3, max_iter=1000)
        proj = ica.fit_transform(data)
        return proj

    def orig(self, data, labels):
        return data

    def compare(self):
        comp_algs = {
                'PCA Distance': self.spca,
                'Cosine Similarity': self.cosine_sim,
                'DBI': self.davies_bouldin_index
                }
        comp_ax = {
                'Original/Original': 'oo', 
                'K-Means/K-Means': 'kk',
                'Original/K-Means': 'ok'
                }
        comp_pro = {
                'None': self.orig,
                'PCA': self.pca,
                'MDA': self.mda,
                'ICA': self.ica
                }

        pfunc = comp_pro[self.proCurr]

        labels = self.labels
        data = pfunc(self.get_data(), labels)

        if 'k' in comp_ax[self.axesCurr]:
            starts = list()
            for lll in set(labels):
                starts.append(np.mean(data[labels==lll, :], 0))
            
            km = KMeans(n_clusters=len(set(labels)), init=np.asarray(starts), n_init=1)
            k_labels = km.fit_predict(data, labels) ## TODO: add ability to choose projection

        if 'kk' == comp_ax[self.axesCurr]:
            l1 = k_labels
            l2 = k_labels
        elif 'ok' == comp_ax[self.axesCurr]:
            l1 = labels
            l2 = k_labels
        elif 'oo' == comp_ax[self.axesCurr]:
            l1 = labels
            l2 = labels

        if data is not None:
            comps = list()
            do_comp = comp_algs[self.alg]
            for li in set(l1):
                cl = list()
                for li2 in set(l2):
                    A = data[l1==li, :]
                    B = data[l2==li2, :]
                    cl.append(do_comp(A, B))
                comps.append(cl)
            return np.asarray(comps), comp_ax[self.axesCurr]
        return None

    def plotting(self, event=None):
        self.alg = self.algList[self.algchoice.GetSelection()]
        self.axesCurr = self.axesList[self.class_choice.GetSelection()]
        self.proCurr = self.proList[self.prochoice.GetSelection()]
        comparison, ax_tip = self.compare()
        if comparison is not None:
            titles = {
                'PCA Distance': 'PCA Cosine Similarity',
                'Cosine Similarity': 'Cosine Similarity',
                'DBI': 'Davies Bouldin Index'
            }
            ax_titles = {
                'oo': ['Original Classes', 'Original Classes'],
                'ok': ['Original Classes', 'K-Means Classes'],
                'kk': ['K-Means Classes', 'K-Means Classes']
            }
            labels = self.labels

            self.fig.clf()

            ax = self.fig.add_subplot(111)
            ax.set_title('Metric:\n' + titles[self.alg] +'\nProjection: %s' % (self.proCurr,), size=7)

            mask = np.tri(comparison.shape[0], k=-1)
            comparison_masked = np.ma.array(comparison.copy(), mask=mask)
            cmap = CM.get_cmap('jet', 10)
            cmap.set_bad('#eeefff')
            p = ax.pcolormesh(comparison_masked, vmin=np.min(comparison), vmax=np.max(comparison))

            ax.set_xticks(np.arange(comparison.shape[1])+0.5, minor=False)
            ax.set_yticks(np.arange(comparison.shape[0])+0.5, minor=False)
            # if 'k' not in ax_tip:
            ax.set_xticklabels(self.waveforms(), minor=False)
            # if 'k' != ax_tip[0]:
            ax.set_yticklabels(self.waveforms(), minor=False)

            plt.setp(ax.get_xticklabels(), fontsize=5)
            plt.setp(ax.get_yticklabels(), fontsize=5)

            ax.set_xlabel(ax_titles[ax_tip][1], size=6)
            ax.set_ylabel(ax_titles[ax_tip][0], size=6)

            for xmaj in ax.xaxis.get_majorticklocs():
              ax.axvline(x=xmaj-0.5,ls='-',c='k')
            for xmin in ax.xaxis.get_minorticklocs():
              ax.axvline(x=xmin-0.5,ls='--',c='k')

            for ymaj in ax.yaxis.get_majorticklocs():
              ax.axhline(y=ymaj-0.5,ls='-',c='k')
            for ymin in ax.yaxis.get_minorticklocs():
              ax.axhline(y=ymin-0.5,ls='--',c='k')

            # np.ma.set_fill_value(comparison_masked, np.nan)
            for cx in range(comparison_masked.shape[0]):
                for cy in range(comparison_masked.shape[0]):
                    if cx >= cy:
                        ax.annotate('%.3f' %\
                                    round(comparison_masked[cy, cx], 3), xy=(cx, cy), 
                                    xytext=(cx+0.25, cy+0.25), fontsize=5, color='w'
                                    )
            
            cbar = self.fig.colorbar(p)
            cbar.ax.tick_params(labelsize=5)

            self.fig.tight_layout()
            self.canvas.draw()

    def save_fig(self, event):
        output_path = os.path.join(self.export_dir, 'Class_sim_'+self.alg+'_'+self.proCurr+'.png')
        self.fig.savefig(rename_out(output_path), dpi=200)

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)

        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(5)
        sizer_1.Add(self.algtitle, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.algchoice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)

        sizer_1.AddSpacer(5)
        sizer_1.Add(self.protitle, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.prochoice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)

        sizer_1.AddSpacer(5)
        sizer_1.Add(self.class_title, 0, wx.ALIGN_CENTER|wx.EXPAND, 1)
        sizer_1.Add(self.class_choice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)

        sizer_1.AddSpacer(7)

        sizer_1.Add(self.save_button, 0, wx.ALIGN_CENTER, 10)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()