from extimports import *
from SL import SL

class MDA(SL):
    """ Multiple Discriminant Analysis Script
    Author: Robert Capps
    Adapted from (Osan 2007)
    """
    def __init__(self, data, labels):
        SL.__init__(self, data, labels)

    def sw(self):
        weights, means, std = self.classStats(self.data, self.labels) # Weights, means, std
        sw_exp = np.zeros((self.nvar, self.nvar, self.nr_classes))
        sw_0 = np.zeros((self.nvar, self.nvar))
        for ii in set(self.labels):
            ii = int(ii)
            diff_array = self.data[self.labels==ii, :] - means[ii-1,:]
            diff_array = np.dot(diff_array.T, diff_array)
            sw_exp[:, :, ii-1] = np.cov(diff_array, rowvar=0)
            sw_0 = sw_0 + sw_exp[:, :, ii-1]
        return sw_0

    def sb(self):
        weights, means, std = self.classStats(self.data, self.labels) # Weights, means, std
        gmeans = np.mean(self.data, 0)
        sb_exp = np.zeros((self.nvar, self.nvar, self.nr_classes))
        sb_0 = np.zeros((self.nvar, self.nvar))
        for ii in set(self.labels):
            ii = int(ii)
            diff_array = means[ii-1, :] - gmeans
            sb_exp[:, :, ii-1] = weights[ii-1] * np.multiply(diff_array.T, diff_array)
            sb_0 = sb_0 + sb_exp[:, :, ii-1]
        return sb_0

    def projection_weights(self, sb, sw, l1=0., l2=0.):
        lambda_1 = l1
        lambda_2 = l2
        sw = (1 - lambda_1)*sw + lambda_1*np.eye(sw.shape[1])
        sb = (1 - lambda_2)*sb + lambda_2*np.eye(sb.shape[1])
        eigvect, eigval = np.linalg.eig(np.linalg.inv(sw)*sb)
        order = np.flipud(eigval.argsort())
        # print(eigval[eigval.argsort()])
        disc = eigvect[order]
        disc = disc[:, :self.nr_classes]
        return disc

    def fit(self, l1=0., l2=0., test_percent=None):
        # self.splitData(self.data, test_percent=test_percent)
        sb = self.sb()
        sw = self.sw()
        disc = self.projection_weights(sb, sw, l1=l1, l2=l2)
        self.output_data = np.dot(self.data, disc)
        # self.y_train = np.dot(self.data, disc)
        # self.y_test = np.dot(self.testData, disc)
        # return self.y_train
    
    def fit_transform(self, l1=0., l2=0., test_percent=None):
        self.fit(l1=l1, l2=l2, test_percent=test_percent)
        # return self.trainingOrder, self.labels, self.y_train, self.testOrder, self.labels, self.y_test
        # return self.order()
        return self.labels, self.output_data