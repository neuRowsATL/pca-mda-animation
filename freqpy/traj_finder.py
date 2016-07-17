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
from utils import get_waveform_names, to_freq, load_data, raster
import matplotlib.patches as patches

plt.close('all')

class TrajectoryFinder:
    def __init__(self, *args, **kwargs):
        self.color_list = ['r', 'g', 'b', 'k', 'w', 'm', 'c']
        self.freq = args[0]
        self.wv = args[1]
        self.total_time = args[2]

        self.options = {
            'labels': None,
            'eps': -0.5,
            'len_check': 15
        }

        for kk in self.options.keys():
            if kk in kwargs.keys():
                self.options[kk] = kwargs[kk]

    def __call__(self):
        pass

    def __kth_diag_indices(self, a, k):
        rows, cols = np.diag_indices_from(a)
        if k < 0:
            return rows[:k], cols[-k:]
        elif k > 0:
            return rows[k:], cols[:-k]
        else:
            return rows, cols

    def __tconsec(self, diag):
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

    def tspace(self):
        return np.linspace(0, self.total_time, self.wv.shape[0])

    def pca(self, plot_=False):
        pca = PCA(n_components=3)
        proj = pca.fit_transform(self.freq)
        self.proj = proj
        if plot_ is True:
            ax = plt.subplot(projection='3d')
            ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2])
            ax.plot(proj[:, 0], proj[:, 1], proj[:, 2])
            plt.show()
        return proj

    def __pca(self):
        pca = PCA(n_components=3)
        proj = pca.fit_transform(self.freq)
        self.proj = proj

    def do_2d(self, plot_=False, dist_=False, norm_=True):
        self.__pca()
        proj = self.proj
        pd = pairwise_distances(proj, metric='l2')
        opd = pd.copy()
        if norm_ is True:
            pd /= np.max(pd) # normalize
        eps = np.mean(pd) + np.std(pd) * self.options['eps']
        bpdw = np.where(pd >= eps)
        gpdw = np.where(pd < eps)
        # print(len(gpdw[0]), len(np.where(pd <= eps)[0]))
        pd[gpdw] = 1
        pd[bpdw] = 0
        np.fill_diagonal(pd, 0)
        if plot_ is True:
            fig = plt.figure()
            ax = plt.subplot()
            ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
            ax.matshow(pd, cmap=plt.cm.gray)
            plt.show()
        if dist_ is True:
            return opd, pd.copy()
        return pd.copy()

    def find_diags(self, plot_=False):
        pd = self.do_2d()
        tconsec = self.__tconsec
        kth_diag_indices = self.__kth_diag_indices
        if plot_ is True: npd = pd.copy()
        orng = pd.shape[0]
        rows = list()
        cols = list()
        for col_offset in range(orng):
            consec = tconsec(pd.diagonal(col_offset))
            for con in consec:
                if len(con) >= self.options['len_check']:
                    rows.append((con[0], con[-1]))
                    cols.append((con[0] + col_offset, con[-1] + col_offset))
                    if plot_ is True:
                        for r in con:
                            c = col_offset + r
                            npd[r, c] = 255
        if plot_ is True:
            fig = plt.figure()
            ax = plt.subplot()
            ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
            ax.matshow(pd, cmap=plt.cm.gray)
            ax.matshow(npd, cmap=plt.cm.jet, alpha=0.9)
            plt.show()
        return rows, cols

    def do_3d(self, oimgpath, smooth='', diags_='self', 
              sparams=(0.1,0.1), outside_diags=(), cratio=1, 
              outside_eps=None, outside_len=None, outside_name=''):
        total_time = self.total_time
        done = list()
        def incheck(orc, nrc, thresh=20):
            out = list()
            for r, c in nrc:
                for rr, cc in orc:
                    for ri in r:
                        for rii in rr:
                            if abs(rii - ri)<=thresh:
                                for ci in c:
                                    for cii in cc:
                                        if abs(cii - ci)<=thresh:
                                            out.append((r, c))
            return out
        if diags_ == 'self':
            rows, cols = self.find_diags()
        elif diags_ == 'other':
            srows, scols = self.find_diags()
            if outside_eps is not None: self.options['eps'] = outside_eps
            if outside_len is not None: self.options['len_check'] = outside_len
            self.__pca()
            proj = self.proj
            rows, cols = outside_diags
            rows = [(r[0]*cratio, r[1]*cratio) for r in rows]
            cols = [(c[0]*cratio, c[1]*cratio) for c in cols]
            rowcols = incheck([tuple(zz) for zz in zip(srows, scols)], 
                            [tuple(nn) for nn in zip(rows, cols)])
        if smooth == 'bezier':
            proj = bezier(proj, res=proj.shape[0], dim=3)
        elif smooth == 'exp':
            sm = ExpSmooth(proj)
            proj = sm.exponential_double(sparams)[0]
        cnt = 0
        comp = np.mean(self.freq.copy(), 1)
        labels = comp.copy()
        labels[comp>=np.mean(comp)+np.std(comp)] = 0
        lwhere = np.where((np.mean(comp)-np.std(comp) < comp) & (comp < np.mean(comp)+np.std(comp)))
        labels[lwhere] = 1
        labels[np.mean(comp)-np.std(comp) >= comp] = 2
        lcols = [self.color_list[int(cix-1)] for cix in labels]
        for r, c in rowcols:
            plt.close('all')
            if True:
                print("Generating Figure #%03d..." %(cnt,))
                fig = plt.figure(figsize=(20, 10))
                gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
                gs.update(hspace=0.5)
                ax = plt.subplot(gs[0], projection='3d', frame_on=True)
                ax2 = plt.subplot(gs[1], frame_on=True)  # waveform
                pc1 = proj[r[0]:r[1], :]
                pc2 = proj[c[0]:c[1], :]
                stitle = "Projection Length:%s\nMin. Length: %s\nEpsilon: %s" %\
                              (str(len(proj)), 
                                str(self.options['len_check']),
                                str(self.options['eps']))
                if outside_name != '': stitle = stitle + '\nDiags from: %s' % (outside_name,)
                if smooth != '': stitle = 'Smoothing: %s\n' % (smooth,) + stitle 
                fig.suptitle(stitle, size=16)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                ax2.set_title("Waveform")
                ax2.set_xlabel('Time (ms)')
                ax2.set_ylabel('Leg Movement')
                # plot waveform
                time_space = np.linspace(0, total_time, self.wv.shape[0])
                ax2.set_xlim([0, self.total_time])
                ax2.plot(time_space, self.wv, color='k', lw=2.0, marker='')
                ts = (total_time / float(self.wv.shape[0]))
                ax2.axvline(r[0] * ts, color='r')
                ax2.axvline(r[1] * ts, color='r')
                ax2.axvline(c[0] * ts, color='b')
                ax2.axvline(c[1] * ts, color='b')
                ax2.fill_between(x=np.linspace(r[0] * ts, c[1] * ts, 
                                            len(time_space[r[0]:c[1]])),
                    y1=-1.95, y2=1.5, facecolor='green', alpha=0.5)
                rows_time = "Period Start Time: [%.2f ms - %.2f ms]" % (r[0] * ts, r[1] * ts)
                cols_time = "Recurrence Time: [%.2f ms - %.2f ms]" % (c[0] * ts, c[1] * ts)
                r_text = ax.text2D(0., -0.25, rows_time,
                                   verticalalignment='bottom', horizontalalignment='left',
                                   color='r', fontsize=12, transform=ax.transAxes, animated=False)
                c_text = ax.text2D(0., -0.3, cols_time,
                                   verticalalignment='bottom', horizontalalignment='left',
                                   color='b', fontsize=12, transform=ax.transAxes, animated=False)
                # plot all data
                ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], alpha=0.8, s=100, marker='o', c=lcols, edgecolors=lcols)
                # plot trajectory
                for p1, p2 in zip(pc1, pc2):
                    ax.plot([p1[0], p2[0]],
                            [p1[1], p2[1]],
                            zs=[p1[2], p2[2]],
                            lw=0.3, marker='', color='g', alpha=1.)
                pc3 = proj[r[1]:c[0], :]
                ax.plot(pc1[:, 0], pc1[:, 1], zs=pc1[:, 2], lw=1.0,
                        marker='.', color='r', alpha=1., markersize=0.1)
                ax.plot(pc2[:, 0], pc2[:, 1], zs=pc2[:, 2], lw=1.0,
                        marker='.', color='b', alpha=1., markersize=0.1)
                ax.plot(pc3[:, 0], pc3[:, 1], zs=pc3[:, 2], lw=1.0,
                        marker='.', color='k', alpha=1., markersize=0.1)
                fig.canvas.draw()
                fname = os.path.join(
                    oimgpath, 'manifold_%03d.png' % (cnt,))
                fig.savefig(fname)
                cnt += 1
        print("Generated %03d figures." %(cnt,))


def distance_to_vertices(proj, corner):
    # calculate distance from each point in proj to vertex
    return np.linalg.norm(proj - corner, axis=1)

def find_radius(ds, bins=100, plot_=False):
    # calculate the radius theshold
    hist, __= np.histogram(ds, bins=bins)
    hx = np.linspace(0, hist.shape[0], hist.shape[0])
    rix = np.where(hist > 0)[0][0]
    if plot_ is True:
        plt.bar(hx, hist, width=1)
        plt.plot(hx, hist)
        plt.axvline(hx[rix+1], color='r')
        plt.show()
    radius = hx[rix+1]
    if radius < ds.min(): radius += (ds.min()/1.3)
    return radius

def consecutive_times(tspace, ixs, ixs_show, verbose=0):
    times = list()
    def cond(ixs_show, i):
        if len(ixs_show) == 1:
            return True
        if i > 0: return True
        else: return False
    for i in range(len(ixs_show)): # Iterate through each set of consecutive pts
        if cond(ixs_show, i):
            start_time, end_time = tspace[ixs[ixs_show[i-1]+1]], tspace[ixs[ixs_show[i]]]
            if verbose > 0: print start_time, end_time
            times.append((start_time, end_time))
    return times

def shade_raster(raxes, cts, colors, cii=0, alpha=0.5):
    for t0, t1 in cts:
        color = colors[int(cii-1)]
        raxes.add_patch(patches.Rectangle((t0, raxes.get_ylim()[0]), t1-t0, raxes.get_ylim()[1],
                         edgecolor=color, fc=color, alpha=alpha, lw=1.))
    return raxes

def label_raster(tf, corners, raxes):
    colors = ['c', 'b', 'g', 'm', 'k', 'r', 'y', '#999999', '#AACCBB']
    cii = 0
    proj = tf.pca(plot_=False) # get projection
    tspace = tf.tspace() # create appropriate time space
    for corner in corners:
        ds = distance_to_vertices(proj, corner) # calculate distance from each point in proj to vertex
        radius = find_radius(ds, plot_=False) # This was calculated by taking a histogram, take the first peak
        ixs = np.where(ds <= radius)[0] # This returns ixs where distance is closer than radius
        ixs_show = np.where(np.diff(ixs) > 1)[0] # This returns the indices of ixs where there is a break in consecutive pts
        cts = consecutive_times(tspace, ixs, ixs_show) # yields start/end times for plotting
        raxes = shade_raster(raxes, cts, colors, cii=cii) # label raster plot
        # print cii, radius, len(ixs)
        cii += 1
    return raxes

def test_finder():
    spikes_dir = os.path.join(os.path.expanduser('~'), "Desktop/SimSpikes")
    dat = ['stimes1.txt', 'stimes2.txt', 'stimes3.txt']
    dat = [os.path.join(spikes_dir, d) for d in dat]
    total_time = 4501.1999
    small_freq = load_data(dat, nr_pts=1e3)
    small_wv = np.sin(np.logspace(0.0, 1.0, small_freq.shape[0]))
    tiny_freq = load_data(dat, nr_pts=500)
    tiny_wv = np.sin(np.logspace(0.0, 1.0, tiny_freq.shape[0]))
    big_freq = load_data(dat, nr_pts=5e3)
    big_wv = np.sin(np.logspace(0.0, 1.0, big_freq.shape[0]))
    
    tf0 = TrajectoryFinder(tiny_freq, tiny_wv, total_time, len_check=150, eps=0.36)
    tf1 = TrajectoryFinder(small_freq, small_wv, total_time, len_check=160/5, eps=-1.35)
    tf2 = TrajectoryFinder(big_freq, big_wv, total_time, len_check=160, eps=-1.35)

    out_dir = os.path.join(os.path.expanduser('~'), "Desktop/SimSpikes")
    
    # proj = tf2.pca(plot_=False)
    # colors = ['c', 'b', 'g', 'm', 'k', 'r', 'y', '#999999', '#AACCBB']
    corners = np.loadtxt(os.path.join(out_dir, 'scorners.txt'))

    ## === REWRITE THIS CODE!!!!
    raxes = raster(load_data(dat, nr_pts=5e3, full=False))
    raxes = label_raster(tf2, corners, raxes)
    plt.show()
    ## === END REWRITING SECTION :)

if __name__ == '__main__':
    test_finder()

