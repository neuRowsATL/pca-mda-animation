""" Multiple Discriminant Analysis Script
    Python 3.5.1
    Author: Robert Capps
    Adapted from (Osan 2007) 
"""

from extimports import *

class MDA:
    def __init__(self, data, labels):
        if data.shape[1] > data.shape[0]:
            data = data.T
        self.data = data
        self.labels = labels
        self.nr_classes = int(max(set(labels)))
        self.nr_repetitions = self.data.shape[0]
        self.nvar = self.data.shape[1]

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

    def splitData(self, data):
        testData = list()
        trainingData = list()
        chosenVals = list()
        rstate = np.random.RandomState(1112)
        for current_class in set(self.labels):
            possible_data = data[self.labels==current_class, :]
            chosen_val = rstate.randint(0, len(possible_data)-1)
            chosenVals.append(chosen_val)
            testData.append(possible_data[chosen_val, :])
        testData = np.array(testData)
        trainingData = np.delete(data, chosenVals, 0)
        self.trainingData = trainingData
        self.trainingLabels = np.delete(self.labels, chosenVals, 0)
        self.testLabels = self.labels[np.array(chosenVals)]
        self.testData = testData

    def sw(self):
        weights, means, std = self.classStats(self.trainingData, self.trainingLabels) # Weights, means, std
        sw_exp = np.zeros((self.nvar, self.nvar, self.nr_classes))
        sw_0 = np.zeros((self.nvar, self.nvar))
        for ii in set(self.labels):
            ii = int(ii)
            diff_array = self.trainingData[self.trainingLabels==ii, :] - means[ii-1,:]
            sw_exp[:, :, ii-1] = np.cov(diff_array.T)
            sw_0 = sw_0 + sw_exp[:, :, ii-1]
        return sw_0

    def sb(self):
        weights, means, std = self.classStats(self.trainingData, self.trainingLabels) # Weights, means, std
        gmeans = np.mean(self.trainingData, 0)
        sb_exp = np.zeros((self.nvar, self.nvar, self.nr_classes))
        sb_0 = np.zeros((self.nvar, self.nvar))
        for ii in set(self.labels):
            ii = int(ii)
            sb_exp[:, :, ii-1] = np.multiply(np.multiply(weights[ii-1], np.subtract(means[ii-1,:], gmeans).T), 
                                             np.subtract(means[ii-1,:], gmeans))
            sb_0 = sb_0 + sb_exp[:, :, ii-1]
        return sb_0

    def projection_weights(self, sb, sw):
        lambda_1 = -0.2
        lambda_2 = -0.2
        sw = (1 - lambda_1)*sw + lambda_1*np.eye(sw.shape[1])
        sb = (1 - lambda_2)*sb + lambda_2*np.eye(sb.shape[1])
        eigvect, eigval = np.linalg.eig(np.linalg.inv(sw)*sb)
        order = np.flipud(eigval.argsort())
        # print(eigval[eigval.argsort()])
        disc = eigvect[order]
        disc = disc[:, 0:self.nr_classes]
        return disc

    def fit(self):
        self.splitData(self.data)
        sb = self.sb()
        sw = self.sw()
        disc = self.projection_weights(sb, sw)
        self.y_train = np.dot(self.trainingData, disc)
        self.y_test = np.dot(self.testData, disc)
        return self.y_train
    
    def fit_transform(self):
        self.fit()
        return self.trainingLabels, self.y_train, self.testLabels, self.y_test
