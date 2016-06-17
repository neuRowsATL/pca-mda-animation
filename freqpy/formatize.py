from extimports import *

class FormatFileNames(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.files = list()
        self.create_buttons()
        self.create_listctrl()
        self.create_text()
        self.create_input()
        self.__do_layout()

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Path")
        self.listCtrl.InsertColumn(1, "File Type")
        self.listCtrl.SetColumnWidth(0, 300)
        self.listCtrl.SetColumnWidth(1, 300)

    def create_buttons(self):
        bold_title_font = wx.Font(24, family=wx.DEFAULT, weight=wx.BOLD, style=wx.NORMAL)
        
        self.res_title = wx.StaticText(self, -1, "Set Resolution")
        self.res_title.SetFont(bold_title_font)
        
        self.rn_title = wx.StaticText(self, -1, "Format Files")
        self.rn_title.SetFont(bold_title_font)
        
        self.data_title1 = wx.StaticText(self, -1, "Spike Times Files:")
        self.data_title2 = wx.StaticText(self, -1, " | e.g. [CBCO-01.txt] -> [2011116D_CBCO-01.txt]", (100,1))
        self.DataButton = wx.Button(self, 0, "Select Spike On/Off Files")
        # self.DataButton.Bind(wx.EVT_BUTTON, self.open_dialog)
        
        self.labels_title1 = wx.StaticText(self, -1, "Labels Files:", (200, 1))
        # self.labels_title2 = wx.StaticText(self, -1, "Typically only one choice from below is necessary:", (200, 1))

        self.l1_title = wx.StaticText(self, -1, " | e.g. [cbco_labels.txt] -> [pdat_labels.txt]", (200, 1))
        self.LabelsButton1 = wx.Button(self, 1, "Select Labels File (integer labels)")
        
        self.l2_title = wx.StaticText(self, -1, " | e.g. load [label_times.txt] and create [pdat_labels.txt]", (200, 1))
        self.LabelsButton2 = wx.Button(self, 2, "Select Labels File (event times)")
        
        self.ResButton = wx.Button(self, -1, "Set Resolution")
        
        buttons = [(self.DataButton, 'Spike Times'), 
                   (self.LabelsButton1, 'Integer Labels'), 
                   (self.LabelsButton2, 'Event Times')
                ]
        
        self.filetypes = ['Spike Times', 'Integer Labels', 'Event Times']

        for b in buttons:
            b[0].Bind(wx.EVT_BUTTON, self.open_dialog)

        self.RenameButton = wx.Button(self, -1, "Format All Files in List.", size=(250, 250))

    def create_text(self):
        self.titler = wx.StaticText(self, -1, 
                                    "Use this tab to help format your data folder before importing."+\
                                    "\nRefer to the readme for more info."+\
                                    "\n** IMPORTANT ** It is HIGHLY recommended that you back up all files first.", 
                                    (800, 1))

    def create_input(self):
        self.t1_title = wx.StaticText(self, -1, "Choose target number of samples: ")
        self.t1 = wx.TextCtrl(self, -1, "1000")

    def open_dialog(self, event):
        filetype = self.filetypes[event.GetId()]
        dialog = wx.FileDialog(self,
                      message="Select Files to Format",
                      style=wx.OPEN|wx.MULTIPLE
                    )
        if dialog.ShowModal() == wx.ID_OK:
            p = dialog.GetPaths()
            dialog.Destroy()
            for fi in p:
                self.listCtrl.Append([fi,filetype])
                self.files.append([fi, filetype])
        return 1

    def rename_all(self, event):

        self.listCtrl.DeleteAllItems()

        spikes = [ff for ff in self.files if ff[1] == self.filetypes[0]]
        integer_labels = [ff for ff in self.files if ff[1] == self.filetypes[1]]
        event_times = [ff for ff in self.files if ff[1] == self.filetypes[2]]

        if len(spikes) > 0:
            rn_spikes = renamer(spikes, self.YEAR, self.MONTH, self.DAY)
            for rs in rn_spikes:
                self.listCtrl.Append([rs, self.filestypes[0]])

        if len(integer_labels) > 0:
            rn_intlab = renamer(integer_labels, type_='int_lab')
            for ri in rn_intlab:
                self.listCtrl.Append([ri, self.filestypes[1]])

        if len(event_times) > 0:
            pass




    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)

        sizer_1.Add(self.titler, 0, wx.ALIGN_LEFT)

        sizer_1.AddSpacer(20)

        sizer_1.Add(self.res_title, 0, wx.ALIGN_CENTER)
        
        hsize_res = wx.BoxSizer(wx.HORIZONTAL)
        sizer_1.Add(self.t1_title, 0, wx.ALIGN_LEFT)
        hsize_res.Add(self.t1, 0, wx.ALIGN_LEFT)
        hsize_res.Add(self.ResButton, 0)
        sizer_1.Add(hsize_res)

        sizer_1.AddSpacer(10)
        sizer_1.Add(wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL, size=(800,15)))
        sizer_1.AddSpacer(15)

        sizer_1.Add(self.rn_title, 0, wx.ALIGN_CENTER)

        sizer_1.AddSpacer(10)

        sizer_1.Add(self.listCtrl, 0, wx.EXPAND|wx.ALL)

        sizer_1.AddSpacer(20)
        
        sizer_1.Add(self.data_title1, 0, wx.ALIGN_LEFT)

        sizer_1.AddSpacer(10)
        
        hsize_dat = wx.BoxSizer(wx.HORIZONTAL)
        hsize_dat.Add(self.DataButton, 1, wx.ALIGN_LEFT)
        hsize_dat.Add(self.data_title2, 1, wx.ALIGN_LEFT)
        sizer_1.Add(hsize_dat)

        sizer_1.AddSpacer(15)
        sizer_1.Add(wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL, size=(800,5)))
        sizer_1.AddSpacer(15)

        sizer_1.Add(self.labels_title1, 0, wx.ALIGN_LEFT)
        # sizer_1.AddSpacer(10)
        # sizer_1.Add(self.labels_title2, 0, wx.ALIGN_LEFT)
        sizer_1.AddSpacer(15)

        hsize_l1 = wx.BoxSizer(wx.HORIZONTAL)
        hsize_l1.Add(self.LabelsButton1, 0, wx.ALIGN_LEFT)
        hsize_l1.Add(self.l1_title, 0, wx.ALIGN_LEFT)
        sizer_1.Add(hsize_l1)

        sizer_1.AddSpacer(10)
        
        hsize_l2 = wx.BoxSizer(wx.HORIZONTAL)
        hsize_l2.Add(self.LabelsButton2, 0, wx.ALIGN_LEFT)
        hsize_l2.Add(self.l2_title, 0, wx.ALIGN_LEFT)
        sizer_1.Add(hsize_l2)

        sizer_1.AddSpacer(20)
        sizer_1.Add(wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL, size=(800,5)))
        sizer_1.AddSpacer(20)
        
        sizer_1.Add(self.RenameButton, 0, wx.ALIGN_CENTER)

        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()