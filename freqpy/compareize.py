from extimports import *

class Compareize(wx.Panel):
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

    def compare(self):
        labels = np.loadtxt(self.labels[0])
        data = self.get_data()
        pclasses = list()
        def do_comp(chosen, pclasses):
            comps = list()
            for i in set(labels):
                curr = pclasses[int(i)-1]
                # spca = np.linalg.norm(np.inner(chosen, pclasses[int(i)-1]))
                # spca = np.sum(np.dot(np.dot(chosen.T, chosen), np.dot(curr.T, curr))) / (np.linalg.norm(chosen) * np.linalg.norm(curr))
                comps.append(spca)
            return comps
        if data is not None:
            norms = list()
            for li in set(labels):
                cnorm = np.linalg.norm(data[labels==li, :])
                norms.append(cnorm)
            comps = list()
            for ix in range(len(norms)):
                comps.append([(np.tanh(np.exp(-abs(norms[ix]-cn)))+1.0)/2.0 for cn in norms])
            return np.asarray(comps)
        return None

    def plotting(self):
        comparison = self.compare()
        # print(len(comparison))
        if comparison is not None:
            labels = np.loadtxt(self.labels[0])
            ax = self.fig.add_subplot(111)
            ax.set_title('Class Similarity')
            p = ax.pcolormesh(comparison)
            ax.set_xticks(np.arange(comparison.shape[1])+0.5, minor=False)
            ax.set_yticks(np.arange(comparison.shape[0])+0.5, minor=False)
            ax.set_xticklabels(self.waveforms(), minor=False)
            ax.set_yticklabels(self.waveforms(), minor=False)
            # ax.set_yticklabels(range(fchanges.shape[0]), minor=False)
            # ax.set_yticklabels(visible=False)
            plt.setp(ax.get_xticklabels(), fontsize=4)
            plt.setp(ax.get_yticklabels(), fontsize=4)
            # ax.get_yaxis().set_ticks([])
            ax.set_xlabel('Class 1')
            ax.set_ylabel('Class 2')
            self.fig.colorbar(p)
            self.canvas.draw()

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(5)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()