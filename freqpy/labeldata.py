from extimports import *

class LabelData(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.dpi = 200
        self.fig = Figure((5.0, 3.0), dpi=self.dpi)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.data_arr = dict()
        self.cond_arr = dict()
        self.lb_arr = list()
        self.t = 0
        self.create_listbox()
        self.__do_layout()

    def load_data(self, filenames):
        data = dict()
        for ii, filename in enumerate(filenames):
            if ii == 0:
                tagname = filename.split('\\')[-1].split('_')[0]
            if filenames[ii-1].split('_')[0] == filenames[ii].split('_')[0]:
                datum = np.loadtxt(filename,skiprows=2)
                data.update({ii: datum})
        self.data_arr.update({tagname: data})
        if self.t == 0:
            self.init_plot()
            self.t += 1
        for ii, k in enumerate(self.data_arr.keys()):
            if ii > 0 and self.data_arr.keys()[ii-1].split('_')[0] != self.data_arr.keys()[ii].split('_')[0]:
                self.lb.InsertItems([self.data_arr.keys()[ii].split('\\')[-1].split('_')[0]],0)
                self.lb_arr.append(self.data_arr.keys()[ii].split('\\')[-1].split('_')[0]) 
            if ii == 0:
                self.lb.InsertItems([self.data_arr.keys()[ii].split('\\')[-1].split('_')[0]],0)
                self.lb_arr.append(self.data_arr.keys()[ii].split('\\')[-1].split('_')[0])

    def load_conditions(self, filenames):
        for filename in filenames:
            conds = np.loadtxt(filename)
            self.cond_arr.update({filename: conds})

    def create_listbox(self):
        sampleList = list()
        for ii, k in enumerate(self.data_arr.keys()):
            if ii > 0 and self.data_arr.keys()[ii-1].split('\\')[-1].split('_')[0] != self.data_arr.keys()[ii].split('\\')[-1].split('_')[0]:
                sampleList.append(self.data_arr.keys()[ii].split('\\')[-1].split('_')[0])
            if ii == 0:
                sampleList.append(self.data_arr.keys()[ii].split('\\')[-1].split('_')[0])
        self.lb = wx.CheckListBox(self, -1, (80, 50), wx.DefaultSize, sampleList)
        self.Bind(wx.EVT_CHECKLISTBOX, self.plot_selected, self.lb)

    def raster(self, event_times_list, color='k'):
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
        self.axes.yaxis.set_ticks(np.arange(0, len(event_times_list.keys()) + 1, 1))
        for ith in event_times_list.keys():
            self.axes.vlines(event_times_list[ith], ith + 0.5, ith + 1.5, color=color, linewidth=0.2)
        self.axes.set_ylim([0.5, len(event_times_list.keys()) + 0.5])

    def get_current(self, keyname):
        init_arr = dict()
        for ii, _ in enumerate(self.data_arr.keys()):
            if self.data_arr.keys()[ii].split('\\')[-1].split('_')[0] == keyname:
                init_arr = self.data_arr[_]
                break
        return init_arr

    def plot_selected(self, event):
        selected = [i for i in range(self.lb.GetCount()) if self.lb.IsChecked(i)]
        selected = [self.lb_arr[i] for i in selected]
        selarr = dict()
        if len(selected) == 1:
            sel = selected[0]
            selarr = self.get_current(sel)
        elif len(selected) > 1:
            for ii, sel in selected:
                selarr[sel] = self.get_current(sel)
        self.raster(selarr, color='k')
        self.canvas.draw()

    def init_plot(self):
        color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        self.axes = self.fig.add_subplot(111)
        self.raster(self.data_arr[self.data_arr.keys()[0]], color='k')
        self.canvas.draw()

    def __do_layout(self):
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        sizer_2.Add(self.lb, 0, wx.ALIGN_CENTER|wx.EXPAND, 5)
        sizer_2.Add(self.canvas, wx.ALIGN_CENTER|wx.EXPAND)
        self.SetSizer(sizer_2)
        sizer_2.Fit(self)
        self.Layout()