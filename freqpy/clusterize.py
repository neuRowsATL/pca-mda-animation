from extimports import *

class Clusterize(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.fig = Figure((5.5, 3.5), dpi=150)
        self.canvas = FigCanvas(self, -1, self.fig)

        self.is_binary = True

        self.labels = None
        self.data = None
        self.waveform_names = None

        self.data_dir = ''
        self.export_dir = ''

        self.save_button = wx.Button(self, -1, "Save Image as PNG", size=(800, 1))
        self.Bind(wx.EVT_BUTTON, self.save_fig, self.save_button)

        self.binary_title = wx.StaticText(self, -1, "Toggle Binary/Graded Plot:", (80, 10))
        
        self.binary_tog = wx.Button(self, -1, "Toggle", size=(800, 1))
        self.Bind(wx.EVT_BUTTON, self.toggle_binary)

        self.threshList = ['mean', 'mean + 1s', 'mean + 2s']

        self.thresh_title = wx.StaticText(self, -1, "Choose a Threshold:", (80, 10))

        self.thresh_choice = wx.Choice(self, -1, (80, 50), wx.DefaultSize, self.threshList)
        self.Bind(wx.EVT_CHOICE, self.plotting, self.thresh_choice)
        self.thresh_choice.SetSelection(1)

        self.__do_layout()

    def toggle_binary(self, event):
        if self.is_binary is True:
            self.is_binary = False
            self.plotting()
            return 1
        self.is_binary = True
        self.plotting()

    def set_inargs(self, intup):
        self.in_args = intup
        # print('clusterize', self.in_args)

    def get_data(self):
        return self.data

    def clustering(self):
        data = self.get_data()
        labels = self.labels
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

    def sort_cluster(self, freqs):
        freqs = (np.tanh(freqs) + 1.0) / 2.0
        freqs = (freqs - np.min(freqs)) / (np.max(freqs) - np.min(freqs))
        chosen_thresh = self.thresh_choice.GetSelection()
        # if chosen_thresh <= 2:  
        thresh = np.mean(freqs) + (np.std(freqs)*chosen_thresh)
        w = np.where(freqs > thresh)
        if self.is_binary:
            freqs[freqs < thresh] = 0
            freqs[freqs >= thresh] = 1
        w = [(w[0][i], w[1][i]) for i in range(len(w[0]))]
        order = list()
        for r in range(freqs.shape[0]):
            cols = [ww[1] for ww in w if ww[0] == r]
            order.append(sum(cols))
        order = np.array(order)
        return np.flipud(freqs[order.argsort(), :])

    def plotting(self, event=None):
        self.fig.clf()
        fchanges = self.sort_cluster(self.clustering())
        if fchanges is not None:
            labels = self.labels

            ax = self.fig.add_subplot(111)
            ax.set_title('Response Clusters')

            ax.set_xticks(np.arange(fchanges.shape[1])+0.5, minor=False)
            ax.set_yticks(np.arange(fchanges.shape[0])+0.5, minor=False)
            ax.set_xticklabels(self.waveform_names[1:], minor=False)
            plt.setp(ax.get_xticklabels(), fontsize=4)
            plt.setp(ax.get_yticklabels(), fontsize=4)
            ax.get_yaxis().set_ticks([])

            ax.set_xlabel('Class')
            ax.set_ylabel('Neuron')

            p = ax.pcolormesh(fchanges)

            if self.is_binary is False:
                self.fig.colorbar(p)
            
            self.canvas.draw()

    def save_fig(self, event):
        self.fig.savefig(self.export_dir+'FreqRespClusters.png')

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(5)
        sizer_1.Add(self.binary_title, 0, wx.ALIGN_CENTER, 0)
        sizer_1.Add(self.binary_tog, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(5)
        sizer_1.Add(self.thresh_title, 0, wx.ALIGN_CENTER, 0)
        sizer_1.Add(self.thresh_choice, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)

        sizer_1.AddSpacer(5)
        
        sizer_1.Add(self.save_button, wx.ALIGN_CENTER)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()