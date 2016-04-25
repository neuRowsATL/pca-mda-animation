from extimports import *

class Clusterize(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.fig = Figure((5.5, 3.5), dpi=150)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.data_dir = ''
        self.labels = list()
        self.save_button = wx.Button(self, -1, "Save Image as PNG", size=(800, 100))
        self.Bind(wx.EVT_BUTTON, self.save_fig, self.save_button)
        self.__do_layout()

    def get_data(self):
        fname = '_normalized_freq.txt'
        dname = [f for f in os.listdir(self.data_dir) if fname in f]
        if len(dname) > 0:
            data = np.loadtxt(self.data_dir+dname[0])
            if data.shape[0] < data.shape[1]: data = data.T
            return data
        return None

    def clustering(self):
        data = self.get_data()
        labels = np.loadtxt(self.labels[0])
        if data is not None:
            freq_changes = list()
            # Assuming label 1 is no_sim
            f0 = np.mean(data[labels==1, :], 0)
            g0 = np.mean(data[labels==1, :])
            for i in set([ll for ll in labels if ll > 1.0]):
                fi = np.mean(data[labels==i, :], 0)
                Ri = abs(fi - f0) / (g0 + f0)
                freq_changes.append(Ri)
            return np.asarray(freq_changes).T
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

    def sort_cluster(self, freqs):
        thresh = 0.16
        # fmap = freqs > thresh
        alist = np.sum(freqs, 1)
        # alist = np.lexsort([freqs[:, i] for i in range(freqs.shape[1])])
        return np.flipud(freqs[alist.argsort(), :])

    def plotting(self):
        fchanges = self.sort_cluster(self.clustering())
        if fchanges is not None:
            labels = np.loadtxt(self.labels[0])
            ax = self.fig.add_subplot(111)
            ax.set_title('Average Change in Frequency')
            p = ax.pcolormesh(fchanges)
            ax.set_xticks(np.arange(fchanges.shape[1])+0.5, minor=False)
            ax.set_yticks(np.arange(fchanges.shape[0])+0.5, minor=False)
            ax.set_xticklabels(self.waveforms()[1:], minor=False)
            # ax.set_yticklabels(range(fchanges.shape[0]), minor=False)
            # ax.set_yticklabels(visible=False)
            plt.setp(ax.get_xticklabels(), fontsize=4)
            plt.setp(ax.get_yticklabels(), fontsize=4)
            # ax.get_yaxis().set_visible(False)
            ax.get_yaxis().set_ticks([])
            ax.set_xlabel('Class')
            ax.set_ylabel('Neuron')
            self.fig.colorbar(p)
            self.canvas.draw()

    def save_fig(self, event):
        self.fig.savefig(self.data_dir.replace('Data','tmp')+'Avg_FreqResp.png')

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(5)
        
        sizer_1.Add(self.save_button, wx.ALIGN_CENTER)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()