import numpy as np
import os
import json
import sys
from waveform_convert import load_waveform, waveform_compress, waveform_convert

def check_platform():
    win_delim = "\\"
    dar_delim = "/"
    if sys.platform[0:3] == "win":
        delim = win_delim
    elif sys.platform[0:3] == "dar":
        delim = dar_delim
    return delim

def get_waveform_names(data_dir=None, fn='waveform_names.json'):
    if fn == 'waveform_names.json' and data_dir is not None:
        with open(os.path.join(data_dir, fn), 'r') as wf:
            waveform_names = json.load(wf)
    else:
        with open(fn, 'r') as wf:
            waveform_names = json.load(wf)
    return [ww[1] for ww in sorted(waveform_names.items(), key=lambda wt: wt[0])]

def average_waveform(wave, nr_pts=1000):
    wave_out = np.zeros((nr_pts, 1))
    for ii in range(nr_pts):
        curr_wave = wave[ii*int((wave.shape[0]/nr_pts)):(ii+1)*int((wave.shape[0]/nr_pts))]
        wave_out[ii] = np.mean(curr_wave)
    return wave_out

def raster(event_times_list, color='k', cond=None, alpha=1.0, axes=None, idxs=None, proj=None, corners=None, mrads=None):
    """
    https://scimusing.wordpress.com/2013/05/06/making-raster-plots-in-python-with-matplotlib/
    Creates a raster plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    colors = ['c', 'b', 'g', 'm', 'w', 'k', 'r', 'y']
    if axes is None:
        axes = plt.subplot()
    axes.tick_params(axis='both', which='major', labelsize=3)
    axes.tick_params(axis='both', which='minor', labelsize=3)
    axes.set_axis_bgcolor('white')
    axes.set_title('Raster Plot', size=10)
    axes.set_xlabel('time (ms)', size=5)
    axes.set_ylabel('Neuron #', size=5)
    axes.tick_params(axis='both', which='major', labelsize=3)
    axes.tick_params(axis='both', which='minor', labelsize=3)
    axes.yaxis.set_ticks(np.arange(0, len(event_times_list) + 1, 1))
    # tspace = np.linspace(0, max([evt for evtt in event_times_list for evt in evtt]), max(idxs))

    print np.where(cond == 0)
    print np.where(cond == 1)
    print np.where(cond == 2)
    print np.where(cond == 3)
    print np.where(cond == 4)
    print np.where(cond == 5)
    print np.where(cond == 6)

    tspace = np.linspace(0, 4501.1999, 5000)
    for ith, trial in enumerate(event_times_list):
        axes.vlines(trial, ith + 0.5, ith + 1.5, color=color, linewidth=0.2, alpha=alpha)
    axes.set_ylim([0.5, len(event_times_list) + 0.5])
    if idxs is not None:
        trange = [0, 0]
        for iix, ix in enumerate(idxs):
            if ix - trange[1] > 1:
                # axes.add_patch(patches.Rectangle((trange[0], 0), ))
                if iix > 0:
                    # print cond[ix]
                    axes.fill_between(x=np.linspace(tspace[trange[0]], tspace[trange[1]], tspace[trange[1]]-tspace[trange[0]]),
                                    y1=axes.get_ylim()[0], y2=axes.get_ylim()[1], facecolor=colors[int(cond[trange[1]]-1)], alpha=.5, edgecolor='')
                    # print "(%.3f, %.3f, %.3f) | d=%.3f (%.3f) %s" % (proj[ix][0], proj[ix][1], proj[ix][2], np.linalg.norm(proj[ix]-corners[int(cond[trange[1]])]), mrads[int(cond[trange[1]])], str(np.linalg.norm(proj[ix]-corners[int(cond[trange[1]])]) <= mrads[int(cond[trange[1]])]))
                    # print "(%.3f, %.3f, %.3f) | d=%s (%.3f) %s" % (proj[ix][0], proj[ix][1], proj[ix][2], str(abs(proj[ix]-corners[int(cond[trange[1]])])), mrads, str(all([aa <= mrads for aa in abs(proj[ix]-corners[int(cond[trange[1]])])])))
                    # print "%.3f" % (tspace[trange[1]])
                trange = [ix, ix]
            else:
                trange[1] = ix
    return axes

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

def load_data(filenames, full=True, nr_pts=1e3, save=False):
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

def tconsec(diag):
    out = list()
    ix = 0
    while ix < len(diag):
        chain = list()
        if ix >= len(diag):
            break
        while diag[ix] == 1:
            chain.append(ix)
            ix += 1
            if ix >= len(diag):
                break
        out.append(chain)
        ix += 1
    return out

def timeline(vals, nr_pts=1000):
    return np.linspace(min([np.min(v) for v in vals]), max([np.max(v) for v in vals]), nr_pts, endpoint=True)

def labeller(data_freq=None, data_list=None, data_files=None, 
            label_times_file=None, label_list=None, nr_pts=1000):
    
    if label_times_file is not None:
        label_list = list()
        with open(label_times_file, 'r') as lf:
            lines = [line for line in lf]
        lines = lines[0].split('\r')
        if len(lines) == 1:
            lines = lines[0].split('\n')
        for line in lines:
            if 'LABEL' not in line:
                label_list.append(line.split(','))
    
    if data_files is not None:
        freq = load_data(data_files)
        time_space = timeline(load_data(data_files, full=False), nr_pts=nr_pts)
    else:
        freq = data_freq
        time_space = timeline(data_list, nr_pts=nr_pts)
    labels = np.ones((nr_pts, 1))

    for l in label_list:
        on_time = float(l[0])
        off_time = float(l[1])
        label = int(l[2])
        labels[np.where((time_space >= on_time) & (time_space <= off_time))] = label
    return freq, labels.ravel().astype(np.uint8)

def renamer(data_files, add_this=None, type_='spikes'):
    if type_ == 'spikes':
        # add_this = str(YEAR)+str(MONTH)+str(DAY)+'D_'
        add_this = str(add_this)
    elif type_ == 'int_lab':
        add_this = 'pdat_labels.txt'
    out = list()
    for f in data_files:
        fn = f.split(check_platform())[-1]
        fp = f.split(check_platform())[0]
        if type_== 'spikes':
            os.rename(f, os.path.join(fp, add_this+fn))
            out.append(os.path.join(fp, add_this+fn))
        elif type_=='int_lab':
            os.rename(f, os.path.join(fp, add_this))
            out.append(os.path.join(fp, add_this))

    return out

def get_settings(settings_path):
    with open(settings_path, 'r') as sf:
        settings = json.load(sf)
    return settings

def save_settings(new_settings, settings_path):
    with open(settings_path, 'w') as sf:
        json.dump(new_settings, sf)

def rename_out(output_path):
    oi = 0
    orig = output_path
    while os.path.exists(output_path):
        output_path = orig.split('.')[0] + str(oi) + '.' + orig.split('.')[1]
        oi += 1
    return output_path