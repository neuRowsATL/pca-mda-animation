from extimports import *

def opener(args):
    data_dir, temp_file = args
    
    of = dict()
    with open(temp_file, 'r') as tf:
        tlines = tf.readlines()
    if len(tlines) == 1: tlines = tlines[0].split('\r')
    for tl in tlines:
        if 'title' in tl:
            of['title'] = tl.split(':')[1].replace('\n','')
        elif 'ax_labels' in tl:
            of['axes_labels'] = eval(tl.split(':')[1].replace('\n', ''))
        elif 'outmoviename' in tl:
            of['out_name'] = tl.split(':')[1].replace('\n', '')
        elif 'DPI' in tl:
            of['dpi'] = int(tl.split(':')[1].replace('\n', ''))
        elif 'labels_name' in tl:
            of['labels'] = np.loadtxt(os.path.join(data_dir, tl.split(':')[1].replace('\n', '')))

    of['data_dir'] = data_dir

    if of['title'] == 'PCA':
        of['data'] = np.loadtxt([os.path.join(of['data_dir'], fi) for fi in os.listdir(of['data_dir']) if 'normalized_freq.txt' in fi][0])
        if of['data'].shape[0] < of['data'].shape[1]: of['data'] = of['data'].T
        pca = PCA(n_components=3)
        of['projected'] = pca.fit_transform(of['data'])

    elif of['title'] == 'ICA':
        of['data'] = np.loadtxt([os.path.join(of['data_dir'],fi) for fi in os.listdir(of['data_dir']) if 'normalized_freq.txt' in fi][0])
        if of['data'].shape[0] < of['data'].shape[1]: of['data'] = of['data'].T
        ica = FastICA(n_components=3)
        of['projected'] = ica.fit_transform(of['data'])

    elif of['title'] == 'MDA':
        of['projected'] = np.loadtxt(os.path.join(of['data_dir'], '_mda_projected.txt'))
        of['labels'] = np.loadtxt(os.path.join(of['data_dir'], '_mda_labels.txt'))

    elif of['title'] == 'K-Means (PCA)':
        of['projected'] = np.loadtxt(os.path.join(of['data_dir'],'_kmeans_projected.txt'))
        of['labels'] = np.loadtxt(os.path.join(of['data_dir'],'_kmeans_labels.txt'))

    return of

def waveforms(folder):
    with open(os.path.join(folder, 'waveform_names.json'), 'r') as wf:
        waveform_names = json.load(wf)
    return list(waveform_names.values())

def init_func(fig, axes, axes2, title_, ax_labels, projected, 
            labels, waveform, data_dir, 
            all_ret=True, color=None, i=None):
    wave_labels = waveforms(data_dir)
    color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
    centers = list()
    classes = list()

    axes.cla()
    plt.setp(axes.get_xticklabels(), fontsize=4)
    plt.setp(axes.get_yticklabels(), fontsize=4)
    plt.setp(axes.get_zticklabels(), fontsize=4)
    
    text_label = "Frame #: %d" % int(0)
    frame_no = axes.text2D(0., -0.3, text_label,
           verticalalignment='bottom', horizontalalignment='left',
           color='b', fontsize=5, transform=axes.transAxes, animated=False)

    axes2.cla()
    plt.setp(axes2.get_xticklabels(), fontsize=4)
    plt.setp(axes2.get_yticklabels(), fontsize=4)
    axes2.set_xticks(np.arange(0, len(labels), 100))
    axes2.set_xticks(np.arange(0, len(labels), 10), minor=True)
    axes2.plot(waveform, color='k', lw=0.5)

    if i != None:
        axes2.axvline(i, color='r')
        frame_no.set_text("Frame #: %d" % int(i))
    for label in set(labels):
        class_proj = projected[labels==label, :]
        center = np.mean(class_proj, 0)
        if color != None:
            if color == color_list[int(label)-1]:
                aa = 1.0
            else:
                aa = 0.25
        elif color == None: aa = 0.25
        curr_class=axes.scatter(center[0], center[1], center[2], 
              marker='o', s=50, edgecolor='k', 
              c=color_list[int(label)-1],
              label=color_list[int(label)-1], alpha=aa)
        idx_where = np.where(labels == label)[0]
        classes.append(curr_class)
        centers.append(center)
    
    if 'K-Means (PCA)' != title_:
        axes.legend(handles=classes, loc=8,
         scatterpoints=1, ncol=len(set(labels)), fontsize=4.5, 
         labels=wave_labels, frameon=False, 
         bbox_to_anchor=(0., -0.46, 1.0, 0.09), mode='expand',
         borderaxespad=0., borderpad=0., labelspacing=5,
         columnspacing=5, handletextpad=0.
         )
    
    axes.set_title(title_, size=10, y=1.0)
    axes.set_xlabel(ax_labels[0],size=5)
    axes.set_ylabel(ax_labels[1],size=5)
    axes.set_zlabel(ax_labels[2],size=5)
    axes.labelpad = 0
    axes.OFFSETTEXTPAD = 0

    axes2.set_title('Waveform', size=5)
    if all_ret:
        return centers, classes, frame_no
    return centers, classes

def save_anim(data_dir, export_dir, res=None):
    t0 = time.time()
    try:
        os.mkdir(os.path.join(export_dir, 'tmp'))
    except Exception:
        pass
    waveform_list = waveforms(data_dir)
    color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
    input_dict = opener([data_dir, os.path.join(data_dir, '_tmp.txt')])
    out_movie = input_dict['out_name']
    projected = input_dict['projected']
    labels = input_dict['labels']
    
    # Delete temp file
    os.remove(os.path.join(data_dir, '_tmp.txt'))

    # interpolation (bezier)
    if projected.shape[0] <= 1000:
        projected = bezier(projected, res=len(labels))
    else:
        projected = exponential(projected, alpha=0.1)
        # projected = gauss_spline(projected, 10)
        # projected = spline_filter(projected, order=5)
        # projected = gaussian_filter(projected, sigma=3.0, mode='reflect')
        # projected = spline_filter(projected, lmbda=20.0)
    
    waveform = np.loadtxt(os.path.join(data_dir, 'waveform.txt'))
    dpi = int(input_dict['dpi'])

    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    gs = gridspec.GridSpec(2, 1, height_ratios=[7, 1])
    gs.update(hspace=1)
    axes = plt.subplot(gs[0], projection='3d', frame_on=True)
    axes2 = plt.subplot(gs[1], frame_on=True) # waveform

    axes2.cla()
    axes2.set_xticks(np.arange(0, len(labels), 100))
    axes2.set_xticks(np.arange(0, len(labels), 10), minor=True)
    axes2.plot(waveform, color='k')
    axes2.axvline(0, color='r')

    plot_args = (fig, axes, axes2,
                 input_dict['title'], input_dict['axes_labels'], 
                 input_dict['projected'], input_dict['labels'],
                 waveform, data_dir)
    centers, classes, frame_no = init_func(*plot_args)
    axes.autoscale_view()
    xlims = axes.get_xlim()
    ylims = axes.get_ylim()
    zlims = axes.get_zlim()

    range_curr = 10
    total_range = np.arange(1, projected.shape[0]-range_curr-1)

    last_pts = [projected[range_curr:range_curr+1, 0], 
                    projected[range_curr:range_curr+1, 1], 
                    projected[range_curr:range_curr+1, 2]]
    last_color = color_list[0]
    
    fig.canvas.blit()
    for i in total_range:
        color = color_list[int(labels[i])-1]
        centers, classes = init_func(*plot_args, all_ret=False, color=color, i=i)
        axes.set_xlim3d(xlims)
        axes.set_ylim3d(ylims)
        axes.set_zlim3d(zlims)
        center = centers[int(labels[i]-1)]
        axes.view_init(elev=30., azim=i)
        curr_projected = projected[i:i+range_curr, :]
        curr_label = [color_list[int(cc)-1] for cc in labels[i:i+range_curr]]
        try:
            x = curr_projected[:, 0]
            y = curr_projected[:, 1]
            z = curr_projected[:, 2]
        except Exception as E:
            print(E)

        last_arr = np.asarray(last_pts)
        curr_xyz = np.asarray([x, y, z])
        acoef = 0.1
        scoef = 0.5
        for start, end in zip(last_arr.T, curr_xyz.T):
            # print start, end
            axes.plot([start[0], end[0]], 
                      [start[1], end[1]], 
                      zs=[start[2], end[2]], 
                      lw=1.0, color=color, label=color, alpha=acoef,
                      markersize=scoef,
                      marker='o')
            scoef += 0.15
            acoef += 0.1

        last_color = color
        last_pts = [x, y, z]
        fig.canvas.draw()
        filename = '__frame%03d.png' % int(i)
        fig.savefig(os.path.join(os.path.join(export_dir, 'tmp'), filename), dpi='figure')
        if sys.platform[0:3] == "win":
            wx.CallAfter(Publisher.sendMessage, "update", str('{0} of {1}'.format(i, len(total_range))))

    crf = 30
    reso = '1280x720'
    if dpi == 150: 
        crf = 25
        reso = '2560x1440'
    elif dpi == 200:
        crf = 20
        reso = '5120x2880'

    output_path = os.path.join(export_dir, out_movie)
    # output_path = output_path.replace('\\', '/')
    def rename_out(output_path):
        oi = 0
        while os.path.exists(output_path):
            output_path = output_path.split('.')[0] + str(oi) + '.' + output_path.split('.')[1]
            oi += 1
        return output_path

    output_path = rename_out(output_path)

    if sys.platform[0:3] == 'win':
        command = 'ffmpeg -framerate 20 -i %s -s:v ' % (os.path.join(os.path.join(export_dir, 'tmp'), '__frame%03d.png'),) + reso + ' -c:v libx264 ' +\
                  '-crf ' + str(crf) + ' -tune animation -pix_fmt yuv420p ' + output_path
    elif sys.platform[0:3] == 'dar':
        command = 'ffmpeg -framerate 20 -i %s -s:v ' % (os.path.join(os.path.join(export_dir, 'tmp'), '__frame%03d.png'),) + reso + ' -c:v libx264 -crf ' + str(crf) +\
                  ' ' + output_path
    
    # return_code = subprocess.call(command, shell=True)
    # return return_code
    # command = "exec " + command
    command = command.split()
    pro = subprocess.Popen(command, shell=True)
    pro.communicate()
    # pro.kill()
    

class SaveThread(Thread):
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.start()

    def run(self):
        return_code = self.func(self.args[0], self.args[1])
        return return_code