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

% Last Modified by GUIDE v2.5 02-Jan-2016 14:24:37

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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get input data
[filename1,filepath1] = uigetfile({'*.txt','ASCII Data Files'}, 'Select Input Data');
cd(filepath1);
datafile = load(filename1,'-ascii');
handles.odat = datafile;
guidata(hObject, handles);

% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1

% Get Plot Ax. Info
contents = cellstr(get(hObject,'String'));
handles.plot_axis = contents{get(hObject, 'Value')}; 
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
handles.plot_axis = '< P1, P2, P3 >';
guidata(hObject, handles);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get Dimensions selected
dims = regexp(handles.plot_axis(regexp(handles.plot_axis, '\d')), '\d');

% Analyze & Plot Data
if strcmp(handles.analysis, 'None (display raw data)')
    handles.plot_dat = handles.odat;
    if length(dims) == 3
        plot3(handles.plot_dat(1, :), handles.plot_dat(2, :), handles.plot_dat(3, :),...
        'Marker', '.', 'LineStyle', 'none');
    elseif length(dims) <= 2
        ax1 = dims(1);
        ax2 = dims(2);
        plot(handles.plot_dat(ax1, :), handles.plot_dat(ax2, :), 'Marker', '.',...
        'LineStyle', 'none');
    end
elseif strcmp(handles.analysis, 'PCA')
    [eigenvectors1, ~] = eig(cov(handles.odat'));
    handles.pdat = eigenvectors1(:, end - 2:end)'*handles.odat;
elseif regexp(handles.analysis, 'PCA + *') == 1
    [handles.pdat, handles.labels] = HCAClass(handles.odat, 4);
    ClusterVis(handles.pdat', handles.labels);
elseif strcmp(handles.analysis, 'MDA')
    % Remus MDA Script
end


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2

% Select Analysis method
contents = cellstr(get(hObject,'String'));
handles.analysis = contents{get(hObject,'Value')};
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

handles.analysis = 'None (display raw data)';
guidata(hObject, handles);
