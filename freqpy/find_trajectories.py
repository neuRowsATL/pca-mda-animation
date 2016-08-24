import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from mpl_toolkits.mplot3d import axes3d
from smoothing import bezier, ExpSmooth
import matplotlib.gridspec as gridspec
import os
import json
import sys
from utils import tconsec, kth_diag_indices, get_waveform_names

plt.close()

if sys.platform[0:3] == 'win':
    raw_freq_high = np.loadtxt(r'C:\Users\Robbie\Desktop\CBCO\_res_1e5\_normalized_freq.txt')
    waveform_high = np.loadtxt(r'C:\Users\Robbie\Desktop\CBCO\_res_1e5\waveform.txt')
    labels_high = np.loadtxt(r'C:\Users\Robbie\Desktop\CBCO\_res_1e5\pdat_labels.txt')

    raw_freq_low = np.loadtxt(r'C:\Users\Robbie\Desktop\CBCO\_res_1e3\_normalized_freq.txt')
    waveform_low = np.loadtxt(r'C:\Users\Robbie\Desktop\CBCO\_res_1e3\waveform.txt')
    labels_low = np.loadtxt(r'C:\Users\Robbie\Desktop\CBCO\_res_1e3\pdat_labels.txt')

    waveform_names = get_waveform_names(r'C:\Users\Robbie\Desktop\CBCO')

    oimgpath = r'C:\Users\Robbie\Desktop\2011-11-06 PDF\CBCO'

if sys.platform[0:3] == 'dar':
    usr_dir = os.path.expanduser('~')

    raw_freq_low = np.loadtxt(os.path.join(usr_dir, r'Desktop/2011-11-06/res1e3/_normalized_freq.txt'))
    waveform_low = np.loadtxt(os.path.join(usr_dir, r'Desktop/2011-11-06/res1e3/waveform.txt'))
    labels_low = np.loadtxt(os.path.join(usr_dir, r'Desktop/2011-11-06/res1e3/pdat_labels.txt'))

    raw_freq_high = np.loadtxt(os.path.join(usr_dir, r'Desktop/2011-11-06/res1e5/_normalized_freq.txt'))
    waveform_high = np.loadtxt(os.path.join(usr_dir, r'Desktop/2011-11-06/res1e5/waveform.txt'))
    labels_high = np.loadtxt(os.path.join(usr_dir, r'Desktop/2011-11-06/res1e5/pdat_labels.txt'))
    
    waveform_names = get_waveform_names(os.path.join(usr_dir, r'Desktop/2011-11-06'))

    oimgpath = os.path.join(usr_dir, r'Desktop/2011-11-06_CBCO_PDF')

ep_ = -1.05
len_check = 15

loadit = True
if loadit is True:
    raw_freqs = [raw_freq_low, raw_freq_high]
    waveforms = [waveform_low, waveform_high]
    labels_l = [labels_low, labels_high]

    plot_dict = dict()
    plot_dict[2] = {0: '', 1: ''}
    plot_dict[3] = {0: '', 1: ''}

    for prim_ix in range(0, 2):
        for curr_lab in [2, 3]:
            labels = labels_l[prim_ix]
            rfo = raw_freqs[prim_ix].copy()
            wvo = waveforms[prim_ix].copy()
            where_lab = np.where(labels==curr_lab)[0]
            raw_freq = raw_freqs[prim_ix][where_lab, :]
            waveform = waveforms[prim_ix][where_lab]

            if raw_freq.shape[0] < raw_freq.shape[1]:
                raw_freq = raw_freq.T

            raw_freq = np.insert(raw_freq, 0, rfo[where_lab.min()-50:where_lab.min()], axis=0)
            raw_freq = np.insert(raw_freq, -1, rfo[where_lab.max():where_lab.max()+50], axis=0)
            waveform = np.insert(waveform, 0, wvo[where_lab.min()-50:where_lab.min()])
            waveform = np.insert(waveform, -1, wvo[where_lab.max():where_lab.max()+50])

            # if prim_ix == 0: len_check = len_check * 100

            pca = PCA(n_components=3)
            proj = pca.fit_transform(raw_freq)
            # plot_dict.update({prim_ix: {curr_lab: {'proj': proj}}})
            plot_dict[curr_lab][prim_ix] = {'proj': proj.tolist()}

            pd = pairwise_distances(proj, metric='l2')
            pd /= np.max(pd)

            # eps_range = np.linspace(-0.6, 1.5, 5)
            # eps_range = [-1.05]
            # len_range = [15]

            # for ep_ in eps_range:
                # for len_check in len_range:
            eps = np.mean(pd) + np.std(pd) * ep_

            bpdw = np.where(pd >= eps)
            gpdw = np.where(pd < eps)

            pd[gpdw] = 1
            pd[bpdw] = 0
            np.fill_diagonal(pd, 0)

            twoD = False

            if twoD is True:
                fig = plt.figure()
                ax = plt.subplot()
                ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
                ax.matshow(pd, cmap=plt.cm.gray)

            len_check = len_check

            # npd = pd.copy()
            orng = pd.shape[0]
            rows = list()
            cols = list()
            for col_offset in range(orng):
                consec = tconsec(pd.diagonal(col_offset))
                for con in consec:
                    if len(con) >= len_check:
                        rows.append((con[0], con[-1]))
                        cols.append((con[0] + col_offset, con[-1] + col_offset))
                        # for r in con:
                        #     c = col_offset + r
                        #     npd[r, c] = 255
            # if twoD is True:
            #     ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
            #     ax.matshow(npd, cmap=plt.cm.jet, alpha=0.9)

            # plot_dict.update({prim_ix: {curr_lab: {'rows': rows}}})
            # plot_dict.update({prim_ix: {curr_lab: {'cols': cols}}})
            # plot_dict.update({prim_ix: {curr_lab: {'wv': waveform.tolist()}}})
            plot_dict[curr_lab][prim_ix].update({'rows': rows})
            plot_dict[curr_lab][prim_ix].update({'cols': cols})
            plot_dict[curr_lab][prim_ix].update({'wv': waveform.tolist()})
            plot_dict[curr_lab][prim_ix].update({'eps': eps})

    with open(os.path.join(oimgpath, 'pdict.json'), 'w') as pf:
        json.dump(plot_dict, pf)

threeD = False
if threeD is True:
    with open(os.path.join(oimgpath, 'pdict.json'), 'r') as pf:
        plot_dict = json.load(pf)

    color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']

    CL = plot_dict['2']
    OL = plot_dict['3']

    CL['0']['proj'] = bezier(np.array(CL['0']['proj'], res=1000, dim=3))
    OL['0']['proj'] = bezier(np.array(CL['0']['proj'], res=1000, dim=3))

    alpha_exp, gamma_exp = (0.1, 0.1)

    smoother = np.array(ExpSmooth(CL['1']['proj']))
    CL['1']['proj'] = smoother.exponential_double(alpha_exp, gamma_exp)[0]

    smoother2 = ExpSmooth(np.array(OL['1']['proj']))
    OL['1']['proj'] = smoother2.exponential_double(alpha_exp, gamma_exp)[0]

    def consec_check(r, c, done):
        if len(done) > 0:
            chk_rng = 10
            checks = list()
            for d in done:
                dcheck = (d[0][0], d[1][0])
                checks.append(
                    any([(r[0], c[0] - iic) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(
                    any([(r[0] - iic, c[0]) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(
                    any([(r[1], c[0] - iic) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(
                    any([(r[1] - iic, c[0]) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(
                    any([(r[0], c[1] - iic) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(
                    any([(r[0] - iic, c[1]) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(
                    any([(r[1], c[1] - iic) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(
                    any([(r[1] - iic, c[1]) == dcheck for iic in range(-chk_rng, chk_rng)]))
            checks.append(
                any([r[1] == c[1] - iic for iic in range(-chk_rng, chk_rng)]))
            checks.append(
                any([r[0] == c[0] - iic for iic in range(-chk_rng, chk_rng)]))
            checks.append(
                any([r[1] == c[0] - iic for iic in range(-chk_rng, chk_rng)]))
            checks.append(
                any([r[0] == c[1] - iic for iic in range(-chk_rng, chk_rng)]))
            if all([c is False for c in checks]):
                return True
            else:
                return False
        return True

    cnt2 = 0

    for condition in ['CL', 'OL']:
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 1,
                               height_ratios=[4, 1])
                               # width_ratios=[1, 1])
        gs.update(hspace=0.5)
        # gs.update(wspace=0.1)
        ax = plt.subplot(gs[0], projection='3d', frame_on=True)
        # axz = plt.subplot(gs[0, 1], projection='3d', frame_on=False)
        ax2 = plt.subplot(gs[1], frame_on=True)  # waveform1
        # ax3 = plt.subplot(gs[2], frame_on=True)  # waveform2

        def init_ax(waveform=None, pc1=None, pc2=None, r=None, c=None, color_=None, res_=None):
            ax.cla()
            ax2.cla()
            # axz.cla()
            # axz.view_init(elev=20., azim=100)

            fig.suptitle("Periodic Manifold (PCA)\n 2011-11-06K: CBCO\n Min. Length: %s\nEpsilon: %s" %
                         (str(len_check), str(round(eps, 2))), size=16)

            ax.set_title("Manifold\n global view")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")

            # axz.set_title("Manifold\n zoomed view")
            # axz.set_xlabel("PC1")
            # axz.set_ylabel("PC2")
            # axz.set_zlabel("PC3")

            ax2.set_title("Waveform")
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Leg Movement')

            total_time = 942299.940
            time_space = np.linspace(0, total_time, waveform.shape[0])
            ax2.set_xlim([0, total_time])

            ax2.plot(time_space, waveform, color='k', lw=2.0, marker='')

            if r is None and c is None:
                ax2.axvline(0, color='r')
                ax2.axvline(0, color='r')
                ax2.axvline(0, color='b')
                ax2.axvline(0, color='b')
            else:
                ts = (total_time / float(waveform.shape[0]))

                ax2.axvline(r[0] * ts, color='r')
                ax2.axvline(r[1] * ts, color='r')

                ax2.axvline(c[0] * ts, color='b')
                ax2.axvline(c[1] * ts, color='b')

                wv_curr = waveform[r[1]:c[0]]
                ax2.fill_between(x=np.linspace(
                    r[1] * ts, c[0] * ts, len(wv_curr)),
                    y1=-1.95, y2=1.5, facecolor='green', alpha=0.5)

                # times
                rows_time = "Period Start Time: [%.2f ms - %.2f ms]" % (r[0] * ts, r[1] * ts)
                cols_time = "Recurrence Time: [%.2f ms - %.2f ms]" % (c[0] * ts, c[1] * ts)
                r_text = ax.text2D(0., -0.25, rows_time,
                                   verticalalignment='bottom', horizontalalignment='left',
                                   color='r', fontsize=12, transform=ax.transAxes, animated=False)
                c_text = ax.text2D(0., -0.3, cols_time,
                                   verticalalignment='bottom', horizontalalignment='left',
                                   color='b', fontsize=12, transform=ax.transAxes, animated=False)

                # plot between lines
                if res == 'low': lsize = 10.0
                else: lsize = 1.0
                for p1, p2 in zip(pc1, pc2):
                    ax.plot([p1[0], p2[0]],
                            [p1[1], p2[1]],
                            zs=[p1[2], p2[2]],
                            lw=0.3*lsize, marker='', color=color_, alpha=1.)
                    # axz.plot([p1[0], p2[0]],
                    #          [p1[1], p2[1]],
                    #          zs=[p1[2], p2[2]],
                    #          lw=0.3, marker='', color='g', alpha=1.)

                # plot orbit start/end traj
                # p3 = proj[r[1]:c[0], :]

                ax.plot(pc1[:, 0], pc1[:, 1], zs=pc1[:, 2], lw=1.0*lsize,
                        marker='.', color='r', alpha=1., markersize=0.1)
                ax.plot(pc2[:, 0], pc2[:, 1], zs=pc2[:, 2], lw=1.0*lsize,
                        marker='.', color='b', alpha=1., markersize=0.1)

                # ax.scatter(p3[:, 0], p3[:, 1], zs=p3[:, 2], marker='.', c=sel_col, alpha=1., s=10)
                # axz.scatter(p3[:, 0], p3[:, 1], zs=p3[:, 2], marker='.', c=sel_col, alpha=1., s=10)
                # ax.plot(p3[:, 0], p3[:, 1], zs=p3[:, 2], marker='.', color='k', alpha=1.)

                # axz.plot(pc1[:, 0], pc1[:, 1], zs=pc1[:, 2], lw=1.0,
                #          marker='.', color='r', alpha=1., markersize=0.1)
                # axz.plot(pc2[:, 0], pc2[:, 1], zs=pc2[:, 2], lw=1.0,
                #          marker='.', color='b', alpha=1., markersize=0.1)

                ax.autoscale_view()
                xlims = ax.get_xlim()
                ylims = ax.get_ylim()
                zlims = ax.get_zlim()

                new_scale = 1e-5  # 0.005
                new_zmin = zlims[0] - np.abs(zlims[0] * new_scale)
                new_zmax = zlims[1] + np.abs(zlims[1] * new_scale)
                zlims = [new_zmin, new_zmax]

                new_xmin = xlims[0] - np.abs(xlims[0] * new_scale)
                new_xmax = xlims[1] + np.abs(xlims[1] * new_scale)
                xlims = [new_xmin, new_xmax]

                new_ymin = ylims[0] - np.abs(ylims[0] * new_scale)
                new_ymax = ylims[1] + np.abs(ylims[1] * new_scale)
                ylims = [new_ymin, new_ymax]
                ax.set_xlim3d(xlims)
                ax.set_ylim3d(ylims)
                ax.set_zlim3d(zlims)
                return xlims, ylims, zlims

        # init_ax()

        if condition == 'CL':
            proj_low = proj_cl_low
            rows_low = rows_cl_low
            cols_low = cols_cl_low
            wav_low = wav_cl_low
            proj_high = proj_cl_high
            rows_high = proj_cl_high
            cols_high = proj_cl_high
            wav_high = wav_cl_high
        elif condition == 'OL':
            proj_low = proj_ol_low
            rows_low = rows_ol_low
            cols_low = cols_ol_low
            wav_low = wav_ol_low
            proj_high = proj_ol_high
            rows_high = rows_ol_high
            cols_high = cols_ol_high
            wav_high = wav_ol_high

        for res in ['low', 'high']:
            if res == 'low':
                color_ = 'c'
                proj = proj_low
                rows = rows_low
                cols = cols_low
                wav = wav_low
            elif res == 'high':
                color_ = 'm'
                proj = proj_high
                rows = rows_high
                cols = cols_high
                wav = wav_high

            wav = np.array(wav)
            for r, c in zip(rows, cols):

                pc1 = proj[r[0]:r[1], :]
                pc2 = proj[c[0]:c[1], :]

                # plot selected
                xlims, ylims, zlims = init_ax(wav, pc1, pc2, r, c, color_, res)

                fig.canvas.draw()
        fname = os.path.join(
            oimgpath, 'manifold_%03d.png' % (cnt2,))
        fig.savefig(fname)

                # Plot all data
                # clab = [color_list[int(cix - 1)] for cix in labels]
                # classes = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                #             alpha=0.01, c=clab, marker='.')

                # axz.set_xlim3d(xlims)
                # axz.set_ylim3d(ylims)
                # axz.set_zlim3d(zlims)
        cnt2 += 1
