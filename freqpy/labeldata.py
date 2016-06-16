from extimports import *

class LabelData(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.dpi = 200
        self.fig = Figure((5.0, 3.0), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.t = 0
        self.data_dir = ''
        self.export_dir = ''
        self.data = None
        self.labels = None
        self.__do_layout()

    def raster(self, event_times_list, color='k', cond=None):
        """
        https://scimusing.wordpress.com/2013/05/06/making-raster-plots-in-python-with-matplotlib/
        Creates a raster plot
        """
        self.axes.cla()
        self.axes.tick_params(axis='both', which='major', labelsize=3)
        self.axes.tick_params(axis='both', which='minor', labelsize=3)
        self.axes.set_axis_bgcolor('white')
        self.axes.set_title('Raster Plot', size=10)
        self.axes.set_xlabel('time (ms)', size=5)
        self.axes.set_ylabel('Neuron #', size=5)
        self.axes.tick_params(axis='both', which='major', labelsize=3)
        self.axes.tick_params(axis='both', which='minor', labelsize=3)
        self.axes.yaxis.set_ticks(np.arange(0, len(event_times_list) + 1, 1))
        # self.axes.xaxis.set_ticks(np.arange(0, max([np.max(ee) for ee in event_times_list])))
        for ith, trial in enumerate(event_times_list):
            self.axes.vlines(trial, ith + 0.5, ith + 1.5, color=color, linewidth=0.2)
        self.axes.set_ylim([0.5, len(event_times_list) + 0.5])

    def plot_selected(self, event):
        plot_data = self.data
        cond_data = self.labels
        self.raster(plot_data, color='k', cond=cond_data)
        self.canvas.draw()

    def init_plot(self):
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        self.axes = self.fig.add_subplot(111)
        plot_data = self.data
        cond_data = self.labels
        self.raster(plot_data, color='k', cond=cond_data)
        self.canvas.draw()

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)

        sizer_1.Add(self.canvas, wx.ALIGN_CENTER)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()