import os

import numpy as np

import wx
import wx.lib.agw.aui as aui

import matplotlib
matplotlib.use('WXAgg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import Toolbar
from mpl_toolkits.mplot3d import Axes3D

from lib_freqpy_data import *

ID_VIEW_STATUSBAR = wx.NewId()
ID_WINDOW_ENVIRONMENT = wx.NewId()
ID_WINDOW_RASTER = wx.NewId()
ID_WINDOW_VIZ = wx.NewId()
ID_WINDOW_TB_CLUSTER = wx.NewId()

ID_TOOLBAR1_A = wx.NewId()
ID_TOOLBAR1_B = wx.NewId()
ID_TOOLBAR1_C = wx.NewId()


class FreqPyGUI(wx.Frame):
    
    def __init__(self, parent, id=-1, title='FreqPy Mainframe',
                 pos=wx.DefaultPosition, size=(1200, 800),
                 style=wx.DEFAULT_FRAME_STYLE):
        
        wx.Frame.__init__(self, parent, id, title, pos, size, style)
        self.StatusBar = self.CreateStatusBar()        
        self.StatusBar.SetStatusText('Initializing...')
        
        self._mgr = aui.AuiManager(self)
        self._mgr.SetManagedWindow(self)
        
        self.panes = {}
        self.import_files = {}

        self.CreateMenuBar()
        #self.CreateToolBars()
        self.CreatePanes()

        
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(aui.EVT_AUI_PANE_CLOSE, self.PaneClosed)
        
        self.StatusBar.SetStatusText('Ready...')
        
    def CreateMenuBar(self):
        ## Set up main menu bar
        mainMenuBar = wx.MenuBar()
    
        menu = wx.Menu()
        m_open = menu.Append(wx.ID_OPEN, "&Open\tCtrl+O", "Open a saved working environment")
        self.Bind(wx.EVT_MENU, self.Open, m_open)
        m_save = menu.Append(wx.ID_SAVE, "&Save\tCtrl+S", "Save current working environment")
        self.Bind(wx.EVT_MENU, self.Save, m_save)
        m_saveAs = menu.Append(wx.ID_SAVEAS, "&Save As...\tCtrl+Shift+S", "Save current working environment")
        self.Bind(wx.EVT_MENU, self.SaveAs, m_saveAs)        
        mainMenuBar.Append(menu, "&File")
    
    
        menu = wx.Menu()
        m_undo = menu.Append(wx.ID_UNDO, "&Undo\tCtrl+Z", "Undo last action")
        self.Bind(wx.EVT_MENU, self.Undo, m_undo)
        m_redo = menu.Append(wx.ID_REDO, "&Redo\tCtrl+Y", "Redo previous action")
        self.Bind(wx.EVT_MENU, self.Redo, m_redo)
        menu.AppendSeparator()
        m_copy = menu.Append(wx.ID_COPY, "&Copy\tCtrl+C", "Copy selection")
        m_cut = menu.Append(wx.ID_CUT, "C&ut\tCtrl+X", "Cut selection")
        m_paste = menu.Append(wx.ID_PASTE, "&Paste\tCtrl+V", "Paste from clipboard")
        mainMenuBar.Append(menu, "&Edit")        
    
    
        menu = wx.Menu()
        m_statusBar = menu.Append(ID_VIEW_STATUSBAR, "Show Status Bar", "Show status bar", kind=wx.ITEM_CHECK)
        menu.Check(m_statusBar.GetId(), True)
        self.Bind(wx.EVT_MENU, self.Statusbar, m_statusBar)
        menu.AppendSeparator()
        m_zoom100 = menu.Append(wx.ID_ZOOM_100, "Zoom 100%\tCtrl+0", "Zoom to 100%")
        self.Bind(wx.EVT_MENU, self.Zoom100, m_zoom100)
        mainMenuBar.Append(menu, "&View")
    
    
        menu = wx.Menu()
        m_workEnv = menu.Append(ID_WINDOW_ENVIRONMENT, "Work Environment", "Show work environment pane", kind=wx.ITEM_CHECK)
        menu.Check(m_workEnv.GetId(), True)
        self.Bind(wx.EVT_MENU, self.toggle_WorkEnv, m_workEnv)
        self.mItem_workEnv = m_workEnv
        
        m_raster = menu.Append(ID_WINDOW_RASTER, "Raster Plots", "Show raster plot pane", kind=wx.ITEM_CHECK)
        menu.Check(m_raster.GetId(), True)
        self.Bind(wx.EVT_MENU, self.toggle_Rasters, m_raster)
        self.mItem_raster = m_raster

        m_dataViz = menu.Append(ID_WINDOW_VIZ, "Data Visualization", "Show visualization pane", kind=wx.ITEM_CHECK)
        menu.Check(m_dataViz.GetId(), True)
        self.Bind(wx.EVT_MENU, self.toggle_DataViz, m_dataViz)
        self.mItem_dataViz = m_dataViz
        menu.AppendSeparator()

        m_tb_cluster = menu.Append(ID_WINDOW_TB_CLUSTER, "Cluster Toolbar", "Show clustering toolbar", kind=wx.ITEM_CHECK)
        menu.Check(m_tb_cluster.GetId(), True)
        self.Bind(wx.EVT_MENU, self.toggle_Cluster, m_tb_cluster)
        self.mItem_cluster = m_tb_cluster
        
        self.m_window = menu
        mainMenuBar.Append(menu, "&Window")
    
    
        menu = wx.Menu()
        mainMenuBar.Append(menu, "&Help")
    
        self.SetMenuBar(mainMenuBar)   
        
        
    def CreateToolBars(self):
        toolbar1 = aui.AuiToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize, aui.AUI_TB_DEFAULT_STYLE)
        
        toolbar1.SetName("Toolbar 1")
        toolbar1.SetToolBitmapSize(wx.Size(16, 16))
        
        tb1_bmp = wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, wx.Size(16,16))
        toolbar1.AddSimpleTool(ID_TOOLBAR1_A, "Check 1", tb1_bmp, aui.ITEM_CHECK)
        toolbar1.AddSimpleTool(ID_TOOLBAR1_B, "Check 2", tb1_bmp, aui.ITEM_CHECK)
        toolbar1.AddSimpleTool(ID_TOOLBAR1_C, "Check 3", tb1_bmp, aui.ITEM_CHECK)
        toolbar1.Realize()
        
        self.tb_cluster = toolbar1
        
        self._mgr.AddPane(toolbar1, aui.AuiPaneInfo().Name("Toolbar 1").Caption("AuiToolbar").ToolbarPane().Top())
        
        
    def CreatePanes(self):
    
        p_Environment = pane_Environment(self, title='Work Environment Pane', size=(300, 800))
        self._mgr.AddPane(p_Environment,
                          aui.AuiPaneInfo().Left().Layer(1).Name('Work Environment').CloseButton(False))
        self.panes['Work Environment'] = p_Environment          
        
        p_DataViz = pane_DataViz(self, title='Data Visualization Pane', size=(500, 400)) #, ax_projection='3d', orientation=wx.HORIZONTAL)
        self._mgr.AddPane(p_DataViz,
                          aui.AuiPaneInfo().Center().Layer(2).Name('Data Visualization').CloseButton(False).Floatable(False))
        self.panes['Data Visualization'] = p_DataViz
    
        p_Raster = pane_Raster(self, title='Raster Plot Pane', size=(500, 200))
        self._mgr.AddPane(p_Raster,
                          aui.AuiPaneInfo().Bottom().Name('Raster Plots').CloseButton(False))
        self.panes['Raster Plot'] = p_Raster    
    
        self._mgr.Update()        

        
    def OnClose(self, e):
        self._mgr.UnInit()
        self.Destroy()
        
        
    def Open(self, e):
        pass
        
    def Save(self, e):
        pass
    
    def SaveAs(self, e):
        pass
    

    def Undo(self, e):
        pass
    
    def Redo(self, e):
        pass
    
    
    def Zoom100(self, e):
        pass    
    
    def Statusbar(self, e):
        pass
    
   
    def toggle_WorkEnv(self, e):
        if self.mItem_workEnv.IsChecked():
            self.p_workEnv.Show()
            #text1 = wx.TextCtrl(self, -1, 'Environment',
                            #wx.DefaultPosition, wx.Size(200,150),
                            #wx.NO_BORDER | wx.TE_MULTILINE)
            #self.p_workEnv = text1
            
            #self._mgr.AddPane(text1, aui.AuiPaneInfo().Left().Name('Work Environment'))
        else:
            self.p_workEnv.Hide()
            #self._mgr.DetachPane(self.p_workEnv)
            
        #self._mgr.Update()
    
    def toggle_Rasters(self, e):
        if self.mItem_raster.IsChecked():
            self.p_raster.Show()
        else:
            self.p_raster.Hide()
            
        #self._mgr.Update()
    
    def toggle_DataViz(self, e):
        if self.mItem_dataViz.IsChecked():
            self.p_dataViz.Show()
        else:
            self.p_dataViz.Hide()
        
        #self._mgr.Update()
    
    def toggle_Cluster(self, e):
        if self.mItem_cluster.IsChecked():
            self.tb_cluster.Show()
        else:
            self.tb_cluster.Hide()
            
        #self._mgr.Update()
        
    def PaneClosed(self, e):
        self.statusbar.SetStatusText("Pane Closed: %s" % e.GetPane().name)
        if e.GetPane().name == 'Work Environment':
            self.m_window.Check(self.mItem_workEnv.GetId(), False)
            

## ===== ===== ===== ===== =====
## ===== ===== ===== ===== =====
                    

class pane_FreqPy(wx.Panel):
    def __init__(self, parent, id=-1, title='FreqPy Pane',
                     pos=wx.DefaultPosition, size=(400, 450),
                     style=wx.DEFAULT_FRAME_STYLE):
            
        self.MainPane = parent
        wx.Panel.__init__(self, parent, id, wx.DefaultPosition, wx.Size(size[0], size[1]))


class pane_plot_FreqPy(pane_FreqPy):
    def __init__(self, parent, id=-1, title='FreqPy Plot Pane',
                 pos=wx.DefaultPosition, size=(400, 450),
                 style=wx.DEFAULT_FRAME_STYLE,
                 orientation=wx.VERTICAL,
                 ax_projection=None):
            
        pane_FreqPy.__init__(self, parent, id=id, title=title, pos=pos, size=size, style=style)  
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
    
        width_ControlPanel = 150
    
        ## Set up BoxSizers and Panels for plotting area
        sizer_plot = wx.BoxSizer(wx.VERTICAL)                
    
        self.fig = Figure()
        self.canvas = FigureCanvas(self, -1, self.fig)        
        self.axes = self.fig.add_axes((0, 0, 1, 1), projection=ax_projection)
    
        self.fig_toolbar = Toolbar(self.canvas)
        self.fig_toolbar.Realize()
    
        sizer_plot.Add(self.canvas, 1, wx.GROW)
        sizer_plot.Add(self.fig_toolbar, 0, wx.GROW)
    
        sizer.Add(sizer_plot, 0, wx.RIGHT, border=10)
    
        ## Set up BoxSizers and Panels for control panel
        sizer_controls = wx.BoxSizer(wx.VERTICAL)
    
        self.ControlPanel = sizer_controls
    
    
        sizer.Add(sizer_controls, wx.EXPAND)
    
        self.SetSizer(sizer)
        self.Layout()
    
        self.canvas.draw()        
        

class pane_Environment(pane_FreqPy):
    
    def __init__(self, parent, id=-1, title='FreqPy Pane',
               pos=wx.DefaultPosition, size=(800,300),
               style=wx.DEFAULT_FRAME_STYLE):
        
        pane_FreqPy.__init__(self, parent, id=-1, title='FreqPy Pane', 
                            pos=wx.DefaultPosition, 
                            size=size, 
                            style=wx.DEFAULT_FRAME_STYLE)
        
        self.filePaths = []
        self.I = Import() # Create an instance of the Import class
        
        #self.notebook = wx.Notebook(self)
        
        
        #self.importFiles = wx.Panel(self.notebook, size=size)
        
        #btn1 = wx.Button(self.importFiles, -1, label='Import selected folders or files')
        #btn1.Bind(wx.EVT_BUTTON, self.DoImport)
        
        #sizer_import = wx.BoxSizer(wx.VERTICAL)
        #sizer_import.Add(wx.StaticText(self.importFiles, label='Drag and drop files or folders to import...'))
        #sizer_import.Add(tab_ImportFiles(self.importFiles))
        
        #sizer_import.Add(btn1)
        
        #self.importFiles.SetSizer(sizer_import)
        
        
        #self.selectFiles = wx.Panel(self.notebook)
        
        #sizer_select = wx.BoxSizer(wx.VERTICAL)
        #sizer_select.Add(tab_SelectFiles(self.selectFiles), 1, wx.EXPAND)
        
        #self.selectFiles.SetSizer(sizer_select)
        
        
        #self.notebook.AddPage(self.importFiles, "Import Files")
        #self.notebook.AddPage(self.selectFiles, "Select Data")
        
        #sizer = wx.BoxSizer(wx.VERTICAL)
        #sizer.Add(self.notebook)
        #self.SetSizer(sizer)
        
        sizer = wx.BoxSizer(wx.VERTICAL)        
        
        sizer.Add(wx.StaticText(self, label='Import file type:'))        
        exts = ['All Files | .*', 'Tab delimited | .txt', 'Comma separated | .csv', 'FreqPy | .fpy',
                'Numpy binary | .npy']
        self.importExts = wx.ComboBox(self, choices=exts)
        self.importExts.Select(0)
        sizer.Add(self.importExts)        

        self.importFiles = wx.Panel(self, size=size)

        sizer.Add(wx.StaticText(self, label='Drag and drop files or folders to import...'))
        sizer.Add(tab_ImportFiles(self.importFiles, self))

        btn1 = wx.Button(self.importFiles, -1, label='Import selected folders or files')
        btn1.Bind(wx.EVT_BUTTON, self.DoImport)
        sizer.Add(btn1)

        self.SetSizer(sizer)
        
    def DoImport(self, e):
        self.MainPane.StatusBar.SetStatusText('Importing %d files' % len(self.filePaths))
        self.I.SetParams({'filenames': self.filePaths}) # Update filenames list
        self.I.ImportData() # Import data
    
class tab_ImportFiles(wx.Panel):
    
    def __init__(self, parent, controls, id=-1, size=(300, 300)):

        wx.Panel.__init__(self, parent, id=id, size=size)
        
        textCtrl = wx.TextCtrl(self, -1, style=wx.TE_MULTILINE)
        textCtrl.SetDropTarget(FileDrop(textCtrl, controls))
        
        sizer = wx.BoxSizer(wx.VERTICAL)        
        sizer.Add(textCtrl, 1, wx.EXPAND)
        self.SetSizer(sizer)
        

class tab_SelectFiles(wx.Panel):
    def __init__(self, parent, id=-1, size=(300,300)):
        
        wx.Panel.__init__(self, parent, id=id, size=size)

        textCtrl = wx.TextCtrl(self, -1, style=wx.TE_MULTILINE)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(textCtrl, 1, wx.EXPAND)
        self.SetSizer(sizer)
        
        
## ===== ===== ===== ===== ===== =====
## http://stackoverflow.com/questions/31391490/wxpython-dragging-a-file-into-window-to-get-file-path

class FileDrop(wx.FileDropTarget):
    def __init__(self, window, controls):
        wx.FileDropTarget.__init__(self)
        self.window = window
        self.controls = controls

    def OnDropFiles(self, x, y, filenames):

        if self.controls.importExts.GetValue().split('.')[1] == '*':
            exts = -1
        else:
            exts = self.controls.importExts.GetValue().split('.')[1]

        for name in filenames:
            try:
                if os.path.isfile(name):
                    fname = os.path.split(name)[1]
                    if exts == -1:
                        self.window.WriteText('\nFile: %s' % fname)
                        self.controls.filePaths.append(name)
                    elif fname.split('.')[1] == exts:
                        self.window.WriteText('\nFile: %s' % fname)
                        self.controls.filePaths.append(name)
                elif os.path.isdir(name):
                    paths = os.listdir(name)
                    for f in paths:
                        if os.path.isfile(os.path.join(name, f)):
                            if f.find('.') > 0:
                                if exts == -1:                                    
                                    self.window.WriteText('\nFile: %s' % f)
                                    self.controls.filePaths.append(os.path.join(name, f))
                                elif f.split('.')[1] == exts:
                                    self.window.WriteText('\nFile: %s' % f)
                                    self.controls.filePaths.append(os.path.join(name, f))
            except IOError, error:
                dlg = wx.MessageDialog(None, 'Error opening file\n' + str(error))
                dlg.ShowModal()
            except UnicodeDecodeError, error:
                dlg = wx.MessageDialog(None, 'Cannot open non ascii files\n' + str(error))
                dlg.ShowModal()

        
## ===== ===== ===== ===== ===== =====
    
    
class pane_Raster(pane_plot_FreqPy):
    pass    


class pane_DataViz(pane_plot_FreqPy):
    def __init__(self, parent, id=-1, title='FreqPy Plot Pane', 
                pos=wx.DefaultPosition, size=(400,450), 
                style=wx.DEFAULT_FRAME_STYLE, 
                orientation=wx.VERTICAL, ax_projection='3d'):
        pane_plot_FreqPy.__init__(self, parent, id=id,
                                  title=title, pos=pos, size=size, style=style,
                                  orientation=orientation, ax_projection=ax_projection)
        
        self.Controls = {}
        
        algNames_dimReduction = ['PCA', 'MDA', 'ICA']
        radBox1 = wx.RadioBox(self, -1, label='Dimensional Reduction', choices=algNames_dimReduction)
        self.Controls['radBox_DimReduction'] = radBox1
        self.ControlPanel.Add(radBox1, wx.TOP, 10)
        
        btn1 = wx.Button(self, -1, label='Run Analysis')
        self.ControlPanel.Add(btn1, wx.TOP, 10)
        btn1.Bind(wx.EVT_BUTTON, self.run_DimensionalReduction)
        
        
        algNames_cluster = ['k-Means', 'GMM']
        radBox2 = wx.RadioBox(self, -1, label='Data Clustering', choices=algNames_cluster)
        self.ControlPanel.Add(radBox2, wx.TOP, 10)
        self.Controls['radBox_Clustering'] = radBox2
        
        btn2 = wx.Button(self, -1, label='Run Cluster')
        self.ControlPanel.Add(btn2, wx.TOP, 10)
        btn2.Bind(wx.EVT_BUTTON, self.run_Clustering)
        
        sb1 = wx.StaticBox(self, label='Visualization Settings')
        sb_sizer = wx.StaticBoxSizer(sb1, orient=wx.VERTICAL)
        
        cb_DataPoints = wx.CheckBox(self, label='Data Points')
        sb_sizer.Add(cb_DataPoints)
        self.Controls['checkBox_DataPoints'] = cb_DataPoints
        
        cb_ClusterCentroids = wx.CheckBox(self, label='Cluster Centroids')
        sb_sizer.Add(cb_ClusterCentroids)
        self.Controls['checkBox_ClusterCentroids'] = cb_ClusterCentroids
        
        self.ControlPanel.Add(sb_sizer)
        
        
        self.Layout()
        
        
    def run_DimensionalReduction(self, e):
        self.MainPane.StatusBar.SetStatusText(self.Controls['radBox_DimReduction'].GetStringSelection())
    
    
    def run_Clustering(self, e):
        self.MainPane.StatusBar.SetStatusText(self.Controls['radBox_Clsutering'].GetStringSelection())
        
        

## ===== ===== ===== ===== =====
## ===== ===== ===== ===== =====

app = wx.App()
frame = FreqPyGUI(None)
frame.Show()
app.MainLoop()
