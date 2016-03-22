""" Multiple Discriminant Analysis Script
    Python 3.5.1
    Author: Robert Capps
    Adapted from (Osan 2007) 
"""

from extimports import *

class MDA:
    def __init__(self, labelled_data):
        self.labelled_data = labelled_data
        self.nr_classes = len(labelled_data.keys())
        def nr_reps():
            count = 0
            arrays = [i for i in itertools.chain.from_iterable(labelled_data.values())]
            for arr in arrays:
                count += 1
            return count
        self.nr_repetitions = nr_reps()
        self.nvar = labelled_data.values()[0].shape[1]

    def classStats(self, data):
        weights = np.array([len(dp) for class_id, dp in data.items()])
        means = np.array([np.mean(dp) for class_id, dp in data.items()])
        std = np.array([np.std(dp) for class_id, dp in data.items()])
        return weights, means, std

    def splitData(self, data):
        trainingData = dict()
        for k, its in data.items():
            trainingData[k] = np.array(random.sample(its, int(len(its)*0.9)))
        testData, _ = DataDiff(data, trainingData)
        self.trainingData = trainingData
        self.testData = testData

    def sw(self):
        weights, means, std = self.classStats(self.trainingData) # Weights, means, std
        sw_exp = np.zeros((self.nvar, self.nvar, self.nr_classes))
        sw_0 = np.zeros((self.nvar, self.nvar))
        labels = list()
        for k, vl in self.trainingData.items():
            for v in vl:
                labels.append(k)
        labels = np.array(labels)
        vals = np.array([i for i in itertools.chain.from_iterable(self.trainingData.values())])
        for k in self.trainingData.keys():
            diff_array = vals[labels == k, :] - means[k-1]
            sw_exp[:, :, k-1] = np.cov(diff_array.T)
            sw_0 = sw_0 + sw_exp[:, :, k-1]
        return sw_0

    def sb(self):
        weights, means, std = self.classStats(self.trainingData) # Weights, means, std
        sb_exp = np.zeros((self.nvar, self.nvar, self.nr_classes))
        sb_0 = np.zeros((self.nvar, self.nvar))
        for k in self.trainingData.keys():
            sb_exp[:, :, k-1] = np.multiply(np.multiply(weights[k-1], means[k-1].T), means[k-1])
            sb_0 = sb_0 + sb_exp[:, :, k-1]
        return sb_0

    def projection_weights(self, sb, sw):
        eigvect, eigval = np.linalg.eig(np.linalg.inv(sw)*sb)
        order = np.flipud(eigval.argsort())
        disc = eigvect[order]
        disc = disc[:, 0:self.nr_classes]
        return disc

    def fit(self):
        self.y_train = list()
        self.y_test = list()
        self.splitData(self.labelled_data)
        sb = self.sb()
        sw = self.sw()
        disc = self.projection_weights(sb, sw)
        for k in self.trainingData.keys():
            self.y_train.append(np.dot(self.trainingData[k], disc))
            self.y_test.append(np.dot(self.testData[k], disc))
        return self.y_train
    
    def fit_transform(self):
        self.fit()
        return self.y_train, self.y_test
