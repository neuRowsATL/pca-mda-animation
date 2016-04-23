from extimports import *

class Clusterize(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.fig = Figure((5.5, 3.5), dpi=150)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.labels = list()
        self.__do_layout()

    def get_data(self):
        fname = '_normalized_freq.txt'
        dname = [f for f in os.listdir('./Data/') if fname in f]
        if len(dname) > 0:
            data = np.loadtxt('./Data/'+dname[0])
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

    def plotting(self):
        fchanges = self.clustering()
        if fchanges is not None:
            labels = np.loadtxt(self.labels[0])
            ax = self.fig.add_subplot(111)
            p = ax.pcolormesh(fchanges)
            ax.set_xticks(np.arange(fchanges.shape[1])+0.5, minor=False)
            ax.set_yticks(np.arange(fchanges.shape[0])+0.5, minor=False)
            ax.set_xticklabels(self.waveforms()[1:], minor=False)
            ax.set_yticklabels(range(fchanges.shape[0]), minor=False)
            plt.setp(ax.get_xticklabels(), fontsize=4)
            plt.setp(ax.get_yticklabels(), fontsize=4)
            ax.set_xlabel('Class')
            ax.set_ylabel('Neuron Number')
            self.fig.colorbar(p)
            self.canvas.draw()

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(5)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()