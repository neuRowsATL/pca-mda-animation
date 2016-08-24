from extimports import *

class SL:
    def __init__(self, data, labels):
        if data.shape[1] > data.shape[0]:
            data = data.T
        labels = labels.ravel()
        self.data = data
        self.labels = labels
        self.nr_classes = int(max(set(labels.flat)))
        self.nr_repetitions = self.data.shape[0]
        self.nvar = self.data.shape[1]
        np.random.seed(0122)

        self.trainingData = None
        self.trainingLabels = None
        self.testData = None
        self.testLabels = None

        self.trainingOrder = None
        self.testOrder = None

    def classStats(self, data, labels):
        classes = set(labels)
        weights = np.array([list(labels).count(c) for c in classes])
        means = np.empty((self.nr_classes, self.nvar))
        stds = np.empty((self.nr_classes, self.nvar))
        for ii in classes:
            ii = int(ii)
            means[ii-1, :] = np.mean(data[labels==ii,:], 0)
            stds[ii-1, :] = np.std(data[labels==ii,:], 0)
        return weights, means, stds

    def splitData(self, data, test_percent=None):
        if test_percent is None: test_percent = 40.0
        if test_percent >= 1.0: test_percent = test_percent / 100.00

        original_indices = np.arange(0, data.shape[0], 1)
        
        X = np.c_[original_indices, data, self.labels]
        
        shuffled_indices = np.random.permutation(X.shape[0])
        X_shuf = X[shuffled_indices, :]

        trainingData = list()
        testData = list()
        
        for lab in set(self.labels):
            where_class = np.where(X_shuf[:, -1]==lab)[0]
            curr = X_shuf[where_class, :]
            curr_percent = int(curr.shape[0] * test_percent)
            training, test = curr[curr_percent:, :], curr[:curr_percent, :]
            trainingData.append(training)
            testData.append(test)

        trainingData = np.concatenate(trainingData)
        testData = np.concatenate(testData)

        self.trainingData = np.array(trainingData[:, 1:-1])
        self.testData = np.array(testData[:, 1:-1])

        self.trainingLabels = np.array(trainingData[:, -1])
        self.testLabels = np.array(testData[:, -1])

        self.trainingOrder = np.array(trainingData[:, 0]).astype(int)
        self.testOrder = np.array(testData[:, 0]).astype(int)

    def order(self):
        all_dat = np.r_[self.trainingData, self.testData]
        all_lab = np.r_[self.trainingLabels, self.testLabels]
        all_ord = np.r_[self.trainingOrder, self.testOrder]

        with_ix = np.c_[all_dat, all_lab, all_ord].tolist()

        original_order = np.array(sorted(with_ix, key=lambda ixk: ixk[-1]))

        LabelsOut = original_order[:, all_dat.shape[1]]
        DataOut = original_order[:, :all_dat.shape[1]]

        # print self.labels, LabelsOut

        return LabelsOut.astype(int), DataOut

    def fit(self, test_percent=None):
        # implement separately for each alg
        self.splitData(self.data, test_percent=test_percent)

    def fit_transform(self, test_percent=None):
        self.fit(test_percent=test_percent)