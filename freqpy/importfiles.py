from extimports import *

class ImportFiles(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.create_buttons()
        self.create_listctrl()
        self.create_radios()
        self.__do_layout()
        self.neurons = list()
        self.conditions = list()

    def create_listctrl(self):
        self.listCtrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)
        self.listCtrl.InsertColumn(0, "File Name")
        self.listCtrl.InsertColumn(1, "File Data Type")
        self.listCtrl.SetColumnWidth(0, 100)
        self.listCtrl.SetColumnWidth(1, 100)

    def create_radios(self):
        self.RadioTitle = wx.StaticText(self, -1, label="Select Data Type:", style=wx.RB_GROUP)
        self.rb1 = wx.RadioButton(self, -1, 'Neural', (10, 10), style=wx.RB_GROUP)
        self.rb2 = wx.RadioButton(self, -1, 'Condition', (10, 30))
        self.Bind(wx.EVT_RADIOBUTTON, self.SetVal, id=self.rb1.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.SetVal, id=self.rb2.GetId())
        self.state = 'Neural'

    def SetVal(self, event):
        state1 = str(self.rb1.GetValue())
        state2 = str(self.rb2.GetValue())
        if state1 == 'True':
            self.state = 'Neural'
        elif state2 == 'True':
            self.state = 'Condition'

    def create_buttons(self):
        self.DataButton = wx.Button(self, -1, "Import File...")
        self.Bind(wx.EVT_BUTTON, self.on_add_file, self.DataButton)

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1.Add(self.listCtrl, 1, wx.EXPAND, 0)
        sizer_1.Add(self.RadioTitle, wx.ALIGN_LEFT)
        sizer_1.Add(self.rb1, wx.ALIGN_LEFT)
        sizer_1.Add(self.rb2, wx.ALIGN_LEFT)
        sizer_1.Add(self.DataButton, 0, wx.ALIGN_LEFT)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()

    def on_add_file(self, event):
        file_choices = "TXT (*.txt)|*.txt"
        dialog = wx.FileDialog(
        self, 
        message="Import file...",
        defaultDir=os.getcwd(),
        defaultFile="",
        wildcard=file_choices,
        style=wx.OPEN|wx.MULTIPLE)
        def multipaths(dialog):
            for d in dialog.GetPaths():
                yield d
        if dialog.ShowModal() == wx.ID_OK:
            for each in multipaths(dialog):
                self.listCtrl.Append([each.split('\\')[-1], 
                                               self.state])
                if self.state == 'Neural':
                    self.neurons.append(each)
                elif self.state == 'Condition':
                    self.conditions.append(each)