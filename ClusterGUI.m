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

% Last Modified by GUIDE v2.5 12-Feb-2016 00:24:51

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

% initialization for later objects
try
    datafile = load('fdat.txt','-ascii');
    handles.odat = datafile;
    set(handles.edit2,'String',{'.\fdat.txt'});
    handles.plot_dat = handles.odat;
catch
end
try
    labelfile = load('pdat_labels.txt','-ascii');
    handles.plot_labels = labelfile;
    set(handles.edit3,'String',{'.\pdat_labels.txt'});
catch
end

handles.thresh = str2double(get(handles.edit5,'String'));

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
set(handles.edit2,'String',{strcat(filepath1, filename1)});
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
cla;
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
handles.plot_axis = '[1, 2, 3]';
dims = eval(handles.plot_axis);
handles.ax1 = dims(1);
handles.ax2 = dims(2);
if numel(dims)>=3
    handles.ax3 = dims(3);
end
% histogram(handles.odat);
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
end

colors = ['r', 'b', 'k', 'm', 'w', 'g', 'c'];

% Analyze & Plot Data
if strcmp(handles.analysis, 'Histogram') == 1
    handles.plot_dat = handles.odat;
    cla;
    axis(handles.axes1);
    histogram(handles.odat);
    axis on;
    xlabel('Data Values');
    ylabel('N');
    drawnow;
    view([0, 90])
elseif strcmp(handles.analysis, 'SEM') == 1
    projected_data = pca(handles.odat, 'NumComponents', 3);
    sem = zeros(max(handles.plot_labels), 3);
    for lab=1:max(handles.plot_labels)
        sem(lab, :) = std(projected_data(handles.plot_labels==lab, :)) / sqrt(numel(handles.plot_labels(handles.plot_labels==lab)));
    end
    cla;
    axis(handles.axes1);
    bar(sem);
    axis on;
    xlabel('Class');
    ylabel('SEM');
    title('Standard Mean Error');
    drawnow;
    view([0, 90])
elseif strcmp(handles.analysis, 'PCA') == 1
%     [eigenvectors1, ~] = eig(cov(handles.odat'));
%     handles.plot_dat = eigenvectors1(:, end - 2:end)'*handles.odat;
    handles.plot_dat = pca(handles.odat, 'NumComponents', 3);
    if length(dims) == 3
        cla;
        xlabel(strcat('P', num2str(handles.ax1)));
        ylabel(strcat('P', num2str(handles.ax2)));
        zlabel(strcat('P', num2str(handles.ax3)));
        hold on;
        for ii=1:length(handles.plot_dat)
            color_num = handles.plot_labels(ii);
            color1 = colors(color_num);
            plot3(handles.plot_dat(1, ii), handles.plot_dat(2, ii), handles.plot_dat(3, ii),...
            'Marker', 'o', 'LineStyle', 'none', 'MarkerFaceColor', color1, 'MarkerEdgeColor', 'b');
        end
        hold off;
        view([30 30 15])
    elseif length(dims) <= 2
        axis(handles.axes1);
        cla;
        hold on;
        for ii=1:length(handles.plot_dat)
            color_num = handles.plot_labels(ii);
            color1 = colors(color_num);
            plot(handles.plot_dat(handles.ax1, ii), handles.plot_dat(handles.ax2, ii),...
            'Marker', '.', 'LineStyle', 'none', 'Color', color1);
        end
        hold off;
        view([0, 90])
        axis on;
        xlabel(strcat('P', num2str(handles.ax1)));
        ylabel(strcat('P', num2str(handles.ax2)));
        drawnow;
    end
elseif strcmp(handles.analysis, 'MDA') == 1
    % Remus MDA Script
    'Please wait... calculating...'
    mda_code;
elseif strcmp(handles.analysis, 'CMA') == 1
    % Robbie's CMA Script
    handles.plot_dat = pca(handles.odat, 3);
    [filename, filepath]=uiputfile('example_movie.avi', 'Save file as...');
    ClusterVis(handles.plot_dat', handles.plot_labels, strcat(filepath, filename), handles.thresh);
end
guidata(hObject, handles);

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
if strcmp(handles.analysis, 'CMA') == 1
    set(handles.edit5, 'Visible','on');
elseif strcmp(handles.analysis, 'CMA') ~= 1
    set(handles.edit5, 'Visible','off');
end
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
handles.plot_labels = [];
handles.analysis = 'PCA';
guidata(hObject, handles);


% --- Executes on button press in labelsPushButton.
function labelsPushButton_Callback(hObject, eventdata, handles)
% hObject    handle to labelsPushButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename1,filepath1] = uigetfile({'*.txt','ASCII Data Files'}, 'Select Labels File');
cd(filepath1);
datafile = load(filename1,'-ascii');
handles.plot_labels = datafile;
handles.labels_path = filepath1;
handles.labels_file = filename1;
guidata(hObject, handles);

% --- Executes on button press in saveButton.
function saveButton_Callback(hObject, eventdata, handles)
% hObject    handle to saveButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename, pathname]= uiputfile({'*.png', 'PNG Image'}, 'Save as...', pwd());
cd(pathname);
saveas(gcf, filename);


% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1


function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double
handles.thresh = str2double(get(hObject,'String'));
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
handles.thresh = str2double(get(hObject,'String'));
set(hObject,'Visible','off');
