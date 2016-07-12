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

class TrajectoryFinder:
    def __init__(self, *args, **kwargs):
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

    @staticmethod
    def kth_diag_indices(a, k):
        rows, cols = np.diag_indices_from(a)
        if k < 0:
            return rows[:k], cols[-k:]
        elif k > 0:
            return rows[k:], cols[:-k]
        else:
            return rows, cols

    @staticmethod
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

    def __pca(self):
        pca = PCA(n_components=3)
        proj = pca.fit_transform(self.freq)
        self.proj = proj

    def do_2d(self, plot_=False):
        self.__pca()
        pd = pairwise_distances(proj, metric='l2')
        pd /= np.max(pd) # normalize
        eps = np.mean(pd) + np.std(pd) * self.options['eps']
        bpdw = np.where(pd >= eps)
        gpdw = np.where(pd < eps)
        pd[gpdw] = 1
        pd[bpdw] = 0
        np.fill_diagonal(pd, 0)
        if plot_ is True:
            fig = plt.figure()
            ax = plt.subplot()
            ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
            ax.matshow(pd, cmap=plt.cm.gray)
            plt.show()
        return pd.copy()

    def find_diags(self, plot_=False):
        pd = self.do_2d()
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

    def do_3d(self, oimgpath):
        done = list()
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
        rows, cols = self.find_diags()
        cnt = 0
        for r, c in zip(rows, cols):
            if consec_check(r, c, done) is True:
                print("Generating Figure #%03d..." %(cnt,))
                fig = plt.figure(figsize=(20, 10))
                gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
                gs.update(hspace=0.5)
                ax = plt.subplot(gs[0], projection='3d', frame_on=True)
                ax2 = plt.subplot(gs[1], frame_on=True)  # waveform
                pc1 = proj[r[0]:r[1], :]
                pc2 = proj[c[0]:c[1], :]
                fig.suptitle("Periodic Manifold (PCA)\nMin. Length: %s\nEpsilon: %s" 
                              % (str(self.options['len_check']),
                                    str(self.options['eps'])), size=16)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                ax2.set_title("Waveform")
                ax2.set_xlabel('Time (ms)')
                ax2.set_ylabel('Leg Movement')
                # plot waveform
                time_space = np.linspace(0, self.total_time, self.wv.shape[0])
                ax2.set_xlim([0, self.total_time])
                ax2.plot(time_space, self.wv, color='k', lw=2.0, marker='')
                ts = (total_time / float(waveform.shape[0]))
                ax2.axvline(r[0] * ts, color='r')
                ax2.axvline(r[1] * ts, color='r')
                ax2.axvline(c[0] * ts, color='b')
                ax2.axvline(c[1] * ts, color='b')
                wv_curr = waveform[r[1]:c[0]]
                ax2.fill_between(x=np.linspace(
                    r[1] * ts, c[0] * ts, len(wv_curr)),
                    y1=-1.95, y2=1.5, facecolor='green', alpha=0.5)
                rows_time = "Period Start Time: [%.2f ms - %.2f ms]" % (r[0] * ts, r[1] * ts)
                cols_time = "Recurrence Time: [%.2f ms - %.2f ms]" % (c[0] * ts, c[1] * ts)
                r_text = ax.text2D(0., -0.25, rows_time,
                                   verticalalignment='bottom', horizontalalignment='left',
                                   color='r', fontsize=12, transform=ax.transAxes, animated=False)
                c_text = ax.text2D(0., -0.3, cols_time,
                                   verticalalignment='bottom', horizontalalignment='left',
                                   color='b', fontsize=12, transform=ax.transAxes, animated=False)
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

        