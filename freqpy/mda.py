""" Multiple Discriminant Analysis Script
    Python 3.5.1
    Author: Robert Capps
    Adapted from work by Dr. Remus Osan (2007) 
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

    @staticmethod
    def classStats(data):
        weights = np.array([len(dp) for class_id, dp in data.items()])
        means = np.array([np.mean(dp, 1) for class_id, dp in data.items()])
        std = np.array([np.std(dp) for class_id, dp in data.items()])
        return weights, means, std

    @staticmethod
    def splitData(data):
        trainingData = dict()
        testData = dict()
        return trainingData, testData

    def fit(self):
        weights, means, std = classStats(self.labelled_data)
        sb_exp = np.multiply(np.multiply(weights, means.T), means)
        sw_exp = np.cov()