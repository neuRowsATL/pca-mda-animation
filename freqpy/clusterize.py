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
        self.prefix = ''

        self.binary_title = wx.StaticText(self, -1, "Toggle Binary/Heat Map:")
        
        self.binary_tog = wx.ToggleButton(self, -1, "Toggle Binary/Heat Map", size=(400, 50))
        self.Bind(wx.EVT_TOGGLEBUTTON, self.toggle_binary)

        self.thresh_title = wx.StaticText(self, -1, r"Choose Threshold: % of (mean + 1std)")

        self.thresh_slider = wx.Slider(self, -1, 100, 0, 100, wx.DefaultPosition, size=(400, 50),
                                       style=wx.SL_HORIZONTAL|wx.SL_AUTOTICKS|wx.SL_LABELS, name="Threshold Slider")

        self.thresh_slider.Bind(wx.EVT_SLIDER, self.plotting)

        self.save_button = wx.Button(self, -1, "Save Image as PNG", size=(400, 50))
        self.Bind(wx.EVT_BUTTON, self.save_fig, self.save_button)

        self.__do_layout()

    def toggle_binary(self, event):
        if self.is_binary is True:
            self.is_binary = False
            self.plotting()
            return 1
        self.is_binary = True
        self.plotting()

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

        chosen_thresh = self.thresh_slider.GetValue() * 1e-2

        thresh = (np.mean(freqs) + np.std(freqs)) * chosen_thresh

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

            ax.set_xlabel('Class', size=8)
            ax.set_ylabel('Neuron', size=8)

            p = ax.pcolormesh(fchanges)

            if self.is_binary is False:
                self.fig.colorbar(p)
            
            self.canvas.draw()

    def save_fig(self, event):
        output_path = os.path.join(self.export_dir, self.prefix+'_FreqRespClusters.png')
        output_path = rename_out(output_path)
        self.fig.savefig(output_path)

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(5)

        sizer_1.Add(self.binary_title, 0, wx.ALIGN_CENTER, 0)

        sizer_1.AddSpacer(5)
        
        hsize_0 = wx.BoxSizer(wx.HORIZONTAL)
        hsize_0.Add(self.binary_tog, 0, wx.ALIGN_CENTER, 0)
        sizer_1.AddSizer(hsize_0, 0, wx.ALIGN_CENTER, 0)

        sizer_1.AddSpacer(5)

        sizer_1.Add(self.thresh_title, 0, wx.ALIGN_CENTER, 0)

        sizer_1.AddSpacer(5)

        hsize = wx.BoxSizer(wx.HORIZONTAL)
        hsize.Add(self.thresh_slider, 0, wx.ALIGN_CENTER, 0)
        sizer_1.AddSizer(hsize, 0, wx.ALIGN_CENTER, 0)

        sizer_1.AddSpacer(5)
        
        hsize_2 = wx.BoxSizer(wx.HORIZONTAL)
        hsize_2.Add(self.save_button, 0, wx.ALIGN_CENTER, 0)
        sizer_1.AddSizer(hsize_2, 0, wx.ALIGN_CENTER, 0)
        
        sizer_1.SetSizeHints(self)
        
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()