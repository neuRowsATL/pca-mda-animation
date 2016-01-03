function varargout = ClusterGUI(varargin)
% CLUSTERGUI MATLAB code for ClusterGUI.fig
%      CLUSTERGUI, by itself, creates a new CLUSTERGUI or raises the existing
%      singleton*.
%
%      H = CLUSTERGUI returns the handle to a new CLUSTERGUI or the handle to
%      the existing singleton*.
%
%      CLUSTERGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CLUSTERGUI.M with the given input arguments.
%
%      CLUSTERGUI('Property','Value',...) creates a new CLUSTERGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ClusterGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ClusterGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ClusterGUI

% Last Modified by GUIDE v2.5 03-Jan-2016 15:44:40

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ClusterGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @ClusterGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ClusterGUI is made visible.
function ClusterGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ClusterGUI (see VARARGIN)

axis vis3d;

% Choose default command line output for ClusterGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ClusterGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ClusterGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in DataPushButton.
function DataPushButton_Callback(hObject, eventdata, handles)
% hObject    handle to DataPushButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get input data
[filename1,filepath1] = uigetfile({'*.txt','ASCII Data Files'}, 'Select Input Data');
cd(filepath1);
datafile = load(filename1,'-ascii');
handles.odat = datafile;
guidata(hObject, handles);

% --- Executes on selection change in AxesMenu.
function AxesMenu_Callback(hObject, eventdata, handles)
% hObject    handle to AxesMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns AxesMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from AxesMenu

% Get Plot Ax. Info
contents = cellstr(get(hObject,'String'));
handles.plot_axis = contents{get(hObject, 'Value')};
dims = eval(handles.plot_axis);
handles.ax1 = dims(1);
handles.ax2 = dims(2);
if length(dims) > 2
    handles.ax3 = dims(3);
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function AxesMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AxesMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
handles.plot_axis = '[1, 2, 3]';
dims = eval(handles.plot_axis);
handles.ax1 = dims(1);
handles.ax2 = dims(2);
handles.ax3 = dims(3);
guidata(hObject, handles);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get Dimensions selected
dims = eval(handles.plot_axis);
handles.ax1 = dims(1);
handles.ax2 = dims(2);
if length(dims) > 2
    handles.ax3 = dims(3);
%     handles.axes1.ZLabel.String = strcat('P', num2str(handles.ax3));
end
% set(handles.axes1.XLabel.String, strcat('P', num2str(handles.ax1)));
% set(handles.axes1.YLabel.String, strcat('P', num2str(handles.ax2)));

% Analyze & Plot Data
if strcmp(handles.analysis, 'None (display raw data)')
    handles.plot_dat = handles.odat;
    if length(dims) == 3
        plot3(handles.plot_dat(1, :), handles.plot_dat(2, :), handles.plot_dat(3, :),...
        'Marker', '.', 'LineStyle', 'none');
    elseif length(dims) <= 2
        axis(handles.axes1);
        plot(handles.plot_dat(handles.ax1, :), handles.plot_dat(handles.ax2, :), 'Marker', '.',...
        'LineStyle', 'none');
        axis on;
        xlabel(strcat('P', num2str(handles.ax1)));
        ylabel(strcat('P', num2str(handles.ax2)));
        drawnow;
    end
elseif strcmp(handles.analysis, 'PCA')
    [eigenvectors1, ~] = eig(cov(handles.odat'));
    handles.plot_dat = eigenvectors1(:, end - 2:end)'*handles.odat;
    if length(dims) == 3
        plot3(handles.plot_dat(1, :), handles.plot_dat(2, :), handles.plot_dat(3, :),...
        'Marker', '.', 'LineStyle', 'none');
    elseif length(dims) <= 2
        axis(handles.axes1);
        plot(handles.plot_dat(handles.ax1, :), handles.plot_dat(handles.ax2, :), 'Marker', '.',...
        'LineStyle', 'none');
        axis on;
        xlabel(strcat('P', num2str(handles.ax1)));
        ylabel(strcat('P', num2str(handles.ax2)));
        drawnow;
    end
elseif regexp(handles.analysis, 'PCA + *') == 1
    [handles.pdat, handles.labels] = HCAClass(handles.odat, 4);
    ClusterVis(handles.pdat', handles.labels);
elseif strcmp(handles.analysis, 'MDA')
    % Remus MDA Script
end


% --- Executes on selection change in AnalysisMethodMenu.
function AnalysisMethodMenu_Callback(hObject, eventdata, handles)
% hObject    handle to AnalysisMethodMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns AnalysisMethodMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from AnalysisMethodMenu

% Select Analysis method
contents = cellstr(get(hObject,'String'));
handles.analysis = contents{get(hObject,'Value')};
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function AnalysisMethodMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AnalysisMethodMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

handles.analysis = 'None (display raw data)';
guidata(hObject, handles);


% --- Executes on button press in labelsPushButton.
function labelsPushButton_Callback(hObject, eventdata, handles)
% hObject    handle to labelsPushButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
