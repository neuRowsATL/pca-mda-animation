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
from matplotlib.colors import cnames as colorlist

raw_freq = np.loadtxt('Data/20120411D_CBCO_normalized_freq.txt')
if raw_freq.shape[0] < raw_freq.shape[1]: raw_freq = raw_freq.T
pca = PCA(n_components=3)
proj = pca.fit_transform(raw_freq)

pd = pairwise_distances(proj, metric='l2')
pd /= np.max(pd)
# opd = pd.copy()

eps = np.mean(pd) + np.std(pd)*(-0.0)

bpdw = np.where(pd >= eps)
gpdw = np.where(pd < eps)

pd[gpdw] = 1
pd[bpdw] = 0
np.fill_diagonal(pd, 0)

# fig = plt.figure()
# ax = plt.subplot()
# ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
# ax.matshow(-pd, cmap=plt.cm.gray)

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

len_check = -50
cnt = 0
npd = pd.copy()
npd.fill(0)
rows = list()
cols = list()
orng = pd.shape[0]
for col_offset in range(orng):
    consec_rows = tconsec(pd.diagonal(col_offset, 1, 0))
    nncol = np.arange(col_offset, col_offset+len(consec_rows)).tolist()
    for cr, ccol in zip(consec_rows, nncol):
        if all([mm >= abs(len_check) for mm in [cr, ccol]]):
            if all([pd[cr+xx, ccol+xx] == 1 for xx in range(len_check, 0, 1)]):
                cnt += 1
                npd[cr, ccol] = 255
                # circle2=plt.Circle((ccol, cr), len_check, color='w', fill=False)
                # fig.gca().add_artist(circle2)


print cnt

# ax = plt.subplot()
# ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
# ax.matshow(npd, cmap=plt.cm.jet, alpha=0.5)

threeD = True
if threeD is True:
    plt.figure()
    labels = np.loadtxt('pdat_labels.txt')
    ax = plt.subplot(projection='3d')

    ii = 0
    for rix in range(npd.shape[0]):
        for cix in range(npd.shape[1]):
            color = colorlist.keys()[ii % len(colorlist.keys())]
            pc1 = proj[rix, :]
            pc2 = proj[cix, :]
            if npd[rix, cix] == 255:
                ax.plot([pc1[0], pc2[0]], 
                        [pc1[1], pc2[1]], 
                        zs=[pc1[2], pc2[2]], 
                        lw=1.0, color=color,marker='o')
                if npd[rix-1,cix-1] == npd[rix+1,cix+1]:
                    ii += 1

    plt.title("DS Orbits Manifold")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.savefig('DS_orbits_manifold.png')

plt.show()



# colorlist = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'w']
# for ri, ci in zip(rows, cols):
#     for rr, cc in zip(ri, ci):
#         pc1 = proj[rr, :]
#         pc2 = proj[cc, :]
#         if not isinstance(rr, int):
#             clab1 = [labels[ric] for ric in rr]
#             color1 = [colorlist[int(lc)-1] for lc in clab1]
#             clab2 = [labels[cic] for cic in cc]
#             color2 = [colorlist[int(lc)-1] for lc in clab2]
#         else:
#             clab1 = labels[rr]
#             color1 = colorlist[int(clab1)-1]
#             clab2 = labels[cc]
#             color2 = colorlist[int(clab2)-1]
#         if len(pc1.shape) < 2:
#             ax.scatter(pc1[0], pc1[1], pc1[2], marker='.', color=color1)
#             ax.scatter(pc2[0], pc2[1], pc2[2], marker='.', color=color2)
#             ax.plot([pc1[0], pc2[0]], 
#                     [pc1[1], pc2[1]], 
#                     zs=[pc1[2], pc2[2]], 
#                     lw=1.0, color='k')
#         else:
#             ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], marker='.', color=color1)
#             ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], marker='.', color=color2)
#             ax.plot([pc1[:, 0], pc2[:, 0]], 
#                     [pc1[:, 1], pc2[:, 1]], 
#                     zs=[pc1[:, 2], pc2[:, 2]], 
#                     lw=1.0, color='k')