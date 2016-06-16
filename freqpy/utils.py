import numpy as np
import os
import json

def get_waveform_names(data_dir):
    with open(os.path.join(data_dir, 'waveform_names.json'), 'r') as wf:
        waveform_names = json.load(wf)
    return list(waveform_names.values())

def average_waveform(wave, nr_pts=1000):
    wave_out = np.zeros((nr_pts, 1))
    for ii in range(nr_pts):
        curr_wave = wave[ii*int((wave.shape[0]/nr_pts)):(ii+1)*int((wave.shape[0]/nr_pts))]
        wave_out[ii] = np.mean(curr_wave)
    return wave_out

def to_freq(data, nr_pts=1e3):
    vals = data
    if vals is not None:
        time_space = np.linspace(min([np.min(v) for v in vals]), max([np.max(v) for v in vals]), nr_pts, endpoint=True)
        delta = time_space[1] - time_space[0]
        time_space = np.insert(time_space, 0, time_space[0] - delta)
        time_space = np.insert(time_space, -1, time_space[-1] + delta)
        freq = np.zeros((len(vals), int(nr_pts)))
        for neuron, datum in enumerate(data):
            for ii in np.arange(nr_pts):
                ii = int(ii)
                count = len(datum[np.where((datum < time_space[ii + 1]) & (datum > time_space[ii]))])
                freq[neuron, ii] = np.divide(count, delta)
        fmean = np.mean(freq, 1)
        fstd = np.std(freq, 1)
        freq = np.array((freq - np.expand_dims(fmean, axis=1)) /
               np.expand_dims(fstd,axis=1))
        freq = (1.000 + np.tanh(freq)) / 2.000
        freq = freq.T
        return freq

def load_data(filenames, full=True, nr_pts=1e3, save=True):
    data = list()
    if len(filenames) > 0:
        for ii, filename in enumerate(filenames):
            try:
                datum = np.loadtxt(filename)
            except ValueError:
                datum = np.loadtxt(filename, skiprows=2)
            data.append(datum)
        if full == True:
            freq = to_freq(data, nr_pts=nr_pts)
            if save == True:
                tag_name = filenames[0].split('-')[0]
                save_name = tag_name + "_normalized_freq.txt"
                np.savetxt(save_name, freq)
            return freq
        return data

def timeline(data, nr_pts=1000):
    return np.linspace(min([np.min(v) for v in vals]), max([np.max(v) for v in vals]), nr_pts, endpoint=True)

def labeller(data_files, label_times_file='label_times.csv', nr_pts=1000):
    label_list = list()
    with open(label_times_file, 'r') as lf:
        lines = [line for line in lf]
    lines = lines[0].split('\r')
    for line in lines:
        if 'LABEL' not in line:
            label_list.append(line.split(','))
    
    freq = load_data(data_files)
    labels = np.ones((nr_pts, 1))
    time_space = timeline(load_data(data_files, full=False), nr_pts=nr_pts)

    for l in label_list:
        on_time = float(l[0])
        off_time = float(l[1])
        label = int(l[2])
        labels[np.where((time_space >= on_time) & (time_space <= off_time))] = label
    np.savetxt('pdat_labels.txt', labels)
    return freq, labels

def renamer(data_files, YEAR, MONTH, DAY):
    add_this = str(YEAR)+str(MONTH)+str(DAY)+'D_'
    for f in data_files:
        os.rename(f, add_this+f)