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

eps = np.mean(pd) + np.std(pd)*(-0.5)

bpdw = np.where(pd >= eps)
gpdw = np.where(pd < eps)

pd[gpdw] = 1
pd[bpdw] = 0
np.fill_diagonal(pd, 0)

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

def consec(condition):
    cw = np.where(np.concatenate(([condition[0]],
                                   condition[:-1] != condition[1:],
                                    [True])))[0]
    cc = np.diff(cw)[::2]
    return cw, cc

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
    # circle2=plt.Circle((ccol, cr), len_check, color='w', fill=False)
    # fig.gca().add_artist(circle2)

print cnt

# ax = plt.subplot()
ax.plot(np.linspace(0, pd.shape[0]), np.linspace(0, pd.shape[0]))
ax.matshow(npd, cmap=plt.cm.jet, alpha=0.9)

threeD = True
if threeD is True:
    plt.figure()
    labels = np.loadtxt('pdat_labels.txt')
    ax = plt.subplot(projection='3d')

    ccc = 0
    colors = ['r', 'b']
    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], alpha=0.05)
    done = list()
    for r, c in zip(rows, cols):
        # if ccc in [18]:
        if ccc >= 0:
            done.append((r, c))
            pc1 = proj[r[0]:r[1], :]
            pc2 = proj[c[0]:c[1], :]
            print r, c
            # print pproj.shape, pproj2.shape
            for p1, p2 in zip(pc1, pc2):
                ax.plot([p1[0], p2[0]], 
                        [p1[1], p2[1]], 
                        zs=[p1[2], p2[2]], 
                        lw=1.0,marker='',color='g',alpha=0.5)
            ax.plot(pc1[:, 0], pc1[:, 1], zs=pc1[:, 2], lw=3.0, marker='o',color='r', alpha=1.)
            ax.plot(pc2[:, 0], pc2[:, 1], zs=pc2[:, 2], lw=3.0, marker='o',color='b', alpha=1.)
        ccc += 1

    # end = False
    # ii = 0
    # prev = 255
    # for rix in range(npd.shape[0]):
    #     for cix in range(npd.shape[1]):
    #         # color = colorlist.keys()[ii % len(colorlist.keys())]
    #         pc1 = proj[rix, :]
    #         pc2 = proj[cix, :]
    #         if npd[rix, cix] == 255:
    #             ax.plot([pc1[0], pc2[0]], 
    #                     [pc1[1], pc2[1]], 
    #                     zs=[pc1[2], pc2[2]], 
    #                     lw=1.0,marker='o')
    #             if prev == npd[rix, cix]:
    #                 prev = npd[rix, cix]
    #                 ii += 1
    #             else:
    #                 ii = 0
    #             if ii == 20:
    #                 ii = 0
    #                 end = True
    #                 break
    #     if end == True:
    #         break

    plt.title("DS Orbits Manifold")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    ax.set_zlabel("PC3")
    # plt.savefig('DS_orbits_manifold.png')

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