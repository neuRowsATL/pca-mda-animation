import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics.pairwise import pairwise_distances
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import gaussian_kde

raw_freq = np.loadtxt('Data/20120411D_CBCO_normalized_freq.txt')
if raw_freq.shape[0] < raw_freq.shape[1]: raw_freq = raw_freq.T
pca = PCA(n_components=3)
proj = pca.fit_transform(raw_freq)

pd = pairwise_distances(proj, metric='l2')
pd /= np.max(pd)
# opd = pd.copy()

eps = np.mean(pd) - np.std(pd)*0.5

bpdw = np.where(pd >= eps)
gpdw = np.where(pd < eps)

pd[gpdw] = 1.0
pd[bpdw] = 0.0

# plt.figure()
# ax = plt.subplot()
# ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
# ax.matshow(pd, cmap=plt.cm.gray)

# print np.diagflat(pd[np.diag_indices_from(pd)])

def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[:k], cols[-k:]
    elif k > 0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols

def consec(condition):
    cw = np.where(np.concatenate(([condition[0]],
                                   condition[:-1] != condition[1:],
                                    [True])))[0]
    cc = np.diff(cw)[::2]
    return cw, cc

def tconsec(diag):
    done = list()
    for di, dd in enumerate(diag):
        if dd == 1 and di not in done:
            xx = 0
            try:
                while diag[di+xx] == dd:
                    done.append(di+xx)
                    xx += 1
            except IndexError:
                continue
    return done

rows = list()
cols = list()
orng = pd.shape[0]-1
for col_offset in range(-orng, orng):
    kd = kth_diag_indices(pd, col_offset)
    consec_rows = tconsec(pd[kd])
    if len(consec_rows) > 0:
        rows.extend(consec_rows)
        if col_offset <= 0:
            cols.extend(np.zeros(len(consec_rows)).tolist())
        else:
            ncol = np.zeros(len(consec_rows))
            ncol.fill(col_offset)
            cols.extend(ncol.tolist())

npd = pd.copy()

npd[rows, cols] = 255

plt.figure()
ax = plt.subplot()
ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
ax.matshow(npd, cmap=plt.cm.jet)

plt.show()