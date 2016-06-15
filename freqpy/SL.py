from extimports import *

class SL:
    def __init__(self, data, labels):
        if data.shape[1] > data.shape[0]:
            data = data.T
        self.data = data
        self.labels = labels
        self.nr_classes = int(max(set(labels)))
        self.nr_repetitions = self.data.shape[0]
        self.nvar = self.data.shape[1]
        np.random.seed(0122)

        self.trainingData = None
        self.trainingLabels = None
        self.testData = None
        self.testLabels = None

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
        trainingData = list()
        testData = list()
        X = np.c_[data, self.labels]
        if test_percent is None: test_percent = 40.0
        if test_percent >= 1.0: test_percent = test_percent / 100.00
        for lab in set(self.labels):
            curr = X[self.labels==lab, :]
            indices = np.random.permutation(curr.shape[0])
            curr_percent = int(len(indices) * test_percent)
            training_idx, test_idx = indices[curr_percent:], indices[:curr_percent]
            training, test = curr[training_idx,:].tolist(), curr[test_idx,:].tolist()
            trainingData.extend(training)
            testData.extend(test)
        trainingData = np.array(trainingData)
        testData = np.array(testData)
        self.trainingData = np.array(trainingData[:, :-1])
        self.testData = np.array(testData[:, :-1])
        self.trainingLabels = np.array(trainingData[:, -1])
        self.testLabels = np.array(testData[:, -1])

    def fit(self, test_percent=None):
        # implement separately for each alg
        self.splitData(self.data, test_percent=test_percent)

    def fit_transform(self, test_percent=None):
        self.fit(test_percent=test_percent)