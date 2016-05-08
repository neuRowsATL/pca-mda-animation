from extimports import *

X = np.loadtxt('./Data/20120411D_normalized_freq.txt')
labels = np.loadtxt('./Data/pdat_labels.txt')
pca = PCA(n_components=3)
Xp = pca.fit_transform(X)
# print Xp.shape
Z = linkage(X, method='average', metric='cosine')
# c, coph_dists = cophenet(Z, pdist(X))
fig = plt.figure()
# ax = fig.add_axes((0, 0, 1, 1), projection='3d')
# ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c='r')
# ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], c='g')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8., labels=labels)


plt.show()