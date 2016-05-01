from extimports import *

class Clusterize(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.fig = Figure((5.5, 3.5), dpi=150)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.data_dir = ''
        self.export_dir = ''
        self.labels = list()
        self.in_args = tuple()
        self.save_button = wx.Button(self, -1, "Save Image as PNG", size=(800, 100))
        self.Bind(wx.EVT_BUTTON, self.save_fig, self.save_button)
        self.__do_layout()

    def set_inargs(self, intup):
        self.in_args = intup
        # print('clusterize', self.in_args)

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
        labels = labels[self.in_args]
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
        thresh = np.mean(freqs) + np.std(freqs)
        w = np.where(freqs > thresh)
        w = [(w[0][i], w[1][i]) for i in range(len(w[0]))]
        order = list()
        for r in range(freqs.shape[0]):
            cols = [ww[1]*10 for ww in w if ww[0] == r]
            order.append(sum(cols))
        order = np.array(order)
        return np.flipud(freqs[order.argsort(), :])

    def plotting(self):
        fchanges = self.sort_cluster(self.clustering())
        if fchanges is not None:
            labels = np.loadtxt(self.labels[0])
            labels = labels[self.in_args]
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