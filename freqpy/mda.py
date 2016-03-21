""" Multiple Discriminant Analysis Script
    Python 3.5.1
    Author: Robert Capps
    Adapted from work by Dr. Remus Osan (2007) 
"""

from extimports import *

class MDA:
    def __init__(self, labelled_data):
        # self.labelled_data = dict([[k, normalize(dp)] for k, dp in labelled_data.items()])
        self.labelled_data = labelled_data
        self.nr_classes = len(labelled_data.keys())
        def nr_reps():
            count = 0
            arrays = [i for i in itertools.chain.from_iterable(labelled_data.values())]
            for arr in arrays:
                count += 1
            return count
        self.nr_repetitions = nr_reps()

    def classStats(self, data):
        weights = np.array([len(dp) for class_id, dp in data.items()])
        means = np.array([np.mean(dp) for class_id, dp in data.items()])
        std = np.array([np.std(dp) for class_id, dp in data.items()])
        return weights, means, std

    def splitData(self, data):
        trainingData = dict()
        for k, its in data.items():
            trainingData[k] = np.array(random.sample(its, int(2*len(its)/3)))
        testData, _ = DataDiff(data, trainingData)
        self.trainingData = trainingData
        self.testData = testData

    def fit(self):
        self.y_train = list()
        self.y_test = list()
        self.splitData(self.labelled_data)
        weights, means, std = self.classStats(self.labelled_data) # Weights, means, std of labelled data
        sb_exp = np.multiply(np.multiply(weights, means.T), means)  # Find sb_exp (weights*means'*means)
        trainingMeans = [np.mean(dp, 1) for class_id, dp in self.trainingData.items()]
        comp = list()
        for k, dp in self.trainingData.items():
            comp.append(dp.T - trainingMeans[k-1])
        sw_exp = list()
        for ci in comp:
            sw_exp.append(np.cov(ci))
        slist = list()
        for sw, sb in zip(sw_exp, sb_exp):
            u, s, v = np.linalg.svd(sw*sb)
            slist.append(s)
        for s, ss in zip(slist, self.trainingData.values()):
            self.y_train.append(np.multiply(ss, s))
        for s, ss in zip(slist, self.testData.values()):
            self.y_test.append(np.multiply(ss, s))
        return self.y_train
    
    def fit_transform(self):
        self.fit()
        return self.y_train, self.y_test
