import numpy as np
from sklearn.decomposition import PCA
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics.pairwise import pairwise_distances
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import gaussian_kde
from smoothing import bezier
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
import os
import json

plt.close()

def waveforms(folder):
    with open(os.path.join(folder, 'waveform_names.json'), 'r') as wf:
        waveform_names = json.load(wf)
    return [ww[1] for ww in sorted(waveform_names.items(), key=lambda wt: wt[0])]

raw_freq = np.loadtxt(r'C:\Users\Robbie\Desktop\Data\20111106K-data\CBCO\_normalized_freq.txt')
waveform = np.loadtxt(r'C:\Users\Robbie\Desktop\Data\20111106K-data\CBCO\waveform.txt')
labels = np.loadtxt(r'C:\Users\Robbie\Desktop\Data\20111106K-data\CBCO\pdat_labels.txt')
waveform_names = waveforms(r'C:\Users\Robbie\Desktop\Data\20111106K-data\CBCO')

if raw_freq.shape[0] < raw_freq.shape[1]: raw_freq = raw_freq.T

pca = PCA(n_components=3)
proj = pca.fit_transform(raw_freq)

pd = pairwise_distances(proj, metric='l2')
pd /= np.max(pd)
# opd = pd.copy()

eps = np.mean(pd) + np.std(pd)*-0.6 # -1.05

bpdw = np.where(pd >= eps)
gpdw = np.where(pd < eps)

pd[gpdw] = 1
pd[bpdw] = 0
np.fill_diagonal(pd, 0)

twoD = False

if twoD == True:
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
    ax.matshow(pd, cmap=plt.cm.gray)

def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[:k], cols[-k:]
    elif k > 0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols

def tconsec(diag):
    out = list()
    ix = 0
    while ix < len(diag):
        chain = list()
        if ix >= len(diag): break
        while diag[ix] == 1:
            chain.append(ix)
            ix += 1
            if ix >= len(diag): break
        out.append(chain)
        ix += 1
    return out

len_check = 15
cnt = 0
npd = pd.copy()
# npd.fill(0)
orng = pd.shape[0]
rows = list()
cols = list()
for col_offset in range(orng):
    consec = tconsec(pd.diagonal(col_offset))
    for con in consec:
        if len(con) >= len_check:
            cnt += 1
            rows.append((con[0], con[-1]))
            cols.append((con[0] + col_offset, con[-1] + col_offset))
            for r in con:
                c = col_offset + r
                npd[r, c] = 255

# print cnt

if twoD == True:
    ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
    ax.matshow(npd, cmap=plt.cm.jet, alpha=0.9)

threeD = True
if threeD is True:
    proj = bezier(proj, res=1000, dim=3)
    imgpath = r'C:\Users\Robbie\Desktop\2011-11-06 PDF IMG'

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1])
    gs.update(hspace=0.5)
    gs.update(wspace=0.1)
    
    ax = plt.subplot(gs[0, 0], projection='3d', frame_on=True)
    axz = plt.subplot(gs[0, 1], projection='3d', frame_on=False)
    ax2 = plt.subplot(gs[1, :], frame_on=True) # waveform

    def init_ax(pc1=None, pc2=None, r=None, c=None):
        ax.cla()
        axz.cla()
        ax2.cla()
        axz.view_init(elev=20., azim=100)

        fig.suptitle("Periodic Manifold (PCA)\n 2011-11-06K: CBCO\n Min. Length: %s" % str(len_check), size=16)

        ax.set_title("Manifold\n global view")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        axz.set_title("Manifold\n zoomed view")
        axz.set_xlabel("PC1")
        axz.set_ylabel("PC2")
        axz.set_zlabel("PC3")

        ax2.set_title("Waveform")
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Leg Movement')

        total_time = 942299.940
        time_space = np.linspace(0, total_time, 1000)
        ax2.plot(time_space, waveform, color='k')
        ax2.set_xlim([0, total_time])
        if r == None and c == None:
            ax2.axvline(0, color='r')
            ax2.axvline(0, color='r')
            ax2.axvline(0, color='b')
            ax2.axvline(0, color='b')
        else:
            ts = (total_time / 1000.0)
            ax2.axvline(r[0]*ts, color='r')
            ax2.axvline(r[1]*ts, color='r')
            wv_r = waveform[r[0]:r[1]]
            trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
            ax2.fill_between(x=np.linspace(r[0], r[1], len(wv_r)), y1=-2, y2=2, facecolor='red', alpha=0.5, transform=trans)

            ax2.axvline(c[0]*ts, color='b')
            ax2.axvline(c[1]*ts, color='b')
            wv_c = waveform[c[0]:c[1]]
            trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
            ax2.fill_between(x=np.linspace(c[0], c[1], len(wv_c)), y1=-2, y2=2, facecolor='blue', alpha=0.5, transform=trans)

            # times
            rows_time = "Period Start Time: [%.2f ms - %.2f ms]" % (r[0]*ts, r[1]*ts)
            cols_time = "Recurrence Time: [%.2f ms - %.2f ms]" % (c[0]*ts, c[1]*ts)
            r_text = ax.text2D(0., -0.25, rows_time,
                   verticalalignment='bottom', horizontalalignment='left',
                   color='r', fontsize=12, transform=ax.transAxes, animated=False)
            c_text = ax.text2D(0., -0.3, cols_time,
                   verticalalignment='bottom', horizontalalignment='left',
                   color='b', fontsize=12, transform=ax.transAxes, animated=False)

            # plot between lines
            for p1, p2 in zip(pc1, pc2):
                ax.plot([p1[0], p2[0]], 
                        [p1[1], p2[1]], 
                        zs=[p1[2], p2[2]], 
                        lw=0.3,marker='',color='g',alpha=1.)
                axz.plot([p1[0], p2[0]], 
                        [p1[1], p2[1]], 
                        zs=[p1[2], p2[2]], 
                        lw=0.3,marker='',color='g',alpha=1.)

            # plot orbit start/end traj
            pc3 = proj[r[1]:c[0], :]
            ax.plot(pc1[:, 0], pc1[:, 1], zs=pc1[:, 2], lw=1.0, marker='.',color='r', alpha=1.)
            ax.plot(pc2[:, 0], pc2[:, 1], zs=pc2[:, 2], lw=1.0, marker='.',color='b', alpha=1.)
            ax.plot(pc3[:, 0], pc3[:, 1], zs=pc3[:, 2], lw=.5, marker='.',color='k', alpha=.8)

            axz.plot(pc3[:, 0], pc3[:, 1], zs=pc3[:, 2], lw=.5, marker='.',color='k', alpha=.8)
            axz.plot(pc1[:, 0], pc1[:, 1], zs=pc1[:, 2], lw=1.0, marker='.',color='r', alpha=1.)
            axz.plot(pc2[:, 0], pc2[:, 1], zs=pc2[:, 2], lw=1.0, marker='.',color='b', alpha=1.)

            ax.autoscale_view()
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            zlims = ax.get_zlim()

            new_scale = 0.005
            new_zmin = zlims[0] - np.abs(zlims[0]*new_scale)
            new_zmax = zlims[1] + np.abs(zlims[1]*new_scale)
            zlims = [new_zmin, new_zmax]

            new_xmin = xlims[0] - np.abs(xlims[0]*new_scale)
            new_xmax = xlims[1] + np.abs(xlims[1]*new_scale)
            xlims = [new_xmin, new_xmax]

            new_ymin = ylims[0] - np.abs(ylims[0]*new_scale)
            new_ymax = ylims[1] + np.abs(ylims[1]*new_scale)
            ylims = [new_ymin, new_ymax]
            return xlims, ylims, zlims

    def consec_check(r, c, done):
        if len(done) > 0:
            chk_rng = 10
            checks = list()
            for d in done:
                dcheck = (d[0][0], d[1][0])
                checks.append(any([(r[0], c[0]-iic) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(any([(r[0]-iic, c[0]) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(any([(r[1], c[0]-iic) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(any([(r[1]-iic, c[0]) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(any([(r[0], c[1]-iic) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(any([(r[0]-iic, c[1]) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(any([(r[1], c[1]-iic) == dcheck for iic in range(-chk_rng, chk_rng)]))
                checks.append(any([(r[1]-iic, c[1]) == dcheck for iic in range(-chk_rng, chk_rng)]))
            checks.append(any([r[1] == c[1]-iic for iic in range(-chk_rng, chk_rng)]))
            checks.append(any([r[0] == c[0]-iic for iic in range(-chk_rng, chk_rng)]))
            checks.append(any([r[1] == c[0]-iic for iic in range(-chk_rng, chk_rng)]))
            checks.append(any([r[0] == c[1]-iic for iic in range(-chk_rng, chk_rng)]))
            if all([c == False for c in checks]):
                # print True
                return True
            else:
                # print False 
                return False
        return True
    
    init_ax()

    ccc = 0
    cnt2 = 0

    done = list()
    for r, c in zip(rows, cols):
        # if ccc in [18]:
        if ccc >= 0:

            if consec_check(r, c, done) == True:
                done.append((r, c))
                pc1 = proj[r[0]:r[1], :]
                pc2 = proj[c[0]:c[1], :]
                
                # plot selected
                xlims, ylims, zlims = init_ax(pc1, pc2, r, c)

                # Plot all data
                color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
                clab = [color_list[int(cix-1)] for cix in labels]
                classes = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], alpha=0.05, c=clab, marker='.')

                axz.set_xlim3d(xlims)
                axz.set_ylim3d(ylims)
                axz.set_zlim3d(zlims)
                
                fig.canvas.draw()

                fname = os.path.join(imgpath, 'manifold_%03d.png' % cnt2)
                fig.savefig(fname)
                cnt2 += 1
                # plt.show()
                # break

        ccc += 1

    print cnt2

# plt.show()