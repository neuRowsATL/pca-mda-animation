import numpy as np
import os
import distance
from joblib import Memory, Parallel, delayed
from tempfile import mkdtemp

class LoadDict(dict):
    # [file type/ext] -> [correct import function]
    def __init__(self):
        self.opt = {
            ('neural', 'txt'): 'NeuralTxt2Np',
            ('mvt', 'txt'): 'MvtTxt2Np'
        }
        super(dict).__init__(dict, self)
        # self.SetVars()

    def __call__(self, filename):
        return CompareName(filename)

    def CompareName(self, filename):
        diffs = list(map(lambda kv: (kv,
                                        distance.nlevenshtein(
                                            seq1='.'.join(kv), 
                                            seq2=filename, method=2), self.opt)))
        return min(diffs, key=lambda itt: itt[1])[0][1]

    # def SetVars(self):
    #     for k, v in self.opt.items():
    #         setattr(self, k, v)

class Data:
    def __init__(self, **kwargs):
        pass

class Import:
    def __init__(self, **kwargs):
        self.Initialize(kwargs)

    def __call__(self, **kwargs):
        self.Initialize(kwargs)
        self.ImportData()

    def Initialize(self, kwargs):
        self.opt = {
            'filenames': list(),
            'folder': '',
            'input_format': 'txt',
            'store_format': 'ndarray',
            'output_format': 'npy',
            'skiprows': 2, # Skips rows for np.loadtxt()
            'resolution': 5e3 # Import resolution
        }
        self._Data = list()
        self.LoadDict = LoadDict()
        self.opt.update(kwargs)
        self.SetVars()

    def SetVars(self):
        for k, v in self.opt.items():
            setattr(self, k, v)

    def IsFolder(self):
        return self.folder != ''

    def IsFilenames(self):
        return len(self.filenames) != 0

    def ConsultLoadDict(self, filename):
        # LoadDict maps file type/ext to correct import function
        filename0 = filename # Keep full path intact
        filename = filename.split(os.sep)[-1] # separate from path
        # Call import function and return
        return getattr(self, self.LoadDict(filename))(filename0)

    def MvtTxt2Np(self, filename):
        """ [Mvt###.txt] -> [np.ndarray] """
        pass

    def NeuralTxt2Np(self, filename):
        """ [Neuron###.txt] -> [np.ndarray] """
        return np.loadtxt(filename, skiprows=self.skiprows)

    def ImportData():
        """ Main Data import function """
        # Check if given list of filenames or a folder
        if self.IsFilenames(): filenames = self.filenames
        elif self.IsFolder() and not self.IsFilenames(): 
            filenames = list(map(lambda fff: os.path.join(self.folder, fff), 
                                        os.listdir(self.folder)))
        if len(filenames) > 0:
            # Iterate through filenames, and attempt to load as self.store_format
            self._Data = list(map(lambda fn: self.ConsultLoadDict(fn), filenames))

    def ToFreq(self):
        data = self._Data
        nr_pts = self.resolution
        def where_pts(nr_pts, time_space, delta, freq, datum):
            for ii in np.arange(nr_pts):
                ii = int(ii)
                count = len(datum[np.where((datum < time_space[ii + 1]) & (datum > time_space[ii]))])
                freq[neuron, ii] = np.divide(count, delta)
        if data is not None:
            freq = np.zeros((len(data), int(nr_pts)))
            time_space = np.linspace(np.array(data).min(), np.array(data).max(), nr_pts)
            delta = time_space[1] - time_space[0]
            time_space = np.insert(time_space, 0, time_space[0] - delta)
            time_space = np.insert(time_space, -1, time_space[-1] + delta)
            for neuron, datum in enumerate(data):
                where_pts(nr_pts, time_space, delta, freq, datum)
            fmean = np.mean(freq, 1)
            fstd = np.std(freq, 1)
            freq = np.array((freq - np.expand_dims(fmean, axis=1)) /
                   np.expand_dims(fstd,axis=1))
            freq = (1.000 + np.tanh(freq)) / 2.000
            freq = freq.T
            return freq

# if __name__ == '__main__':
#     import time
#     narr = [np.random.random((int(1e6),1)) for i in range(33)]
#     t0=time.time()
#     I = Import()
#     print I.ToFreq(narr, 1000).shape
#     print time.time() - t0