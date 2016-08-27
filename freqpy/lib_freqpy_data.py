import numpy as np
import os
import distance
from tempfile import NamedTemporaryFile

class Import:

    """ Import Class
    
    Typical usage:
    
        >> I = Import(filenames=list_of_filenames,
                      resolution=1e5,
                      skiprows=6) # Instantiates

        >> I.ImportData() # Imports data

    ...or using the call function:

       >> I = Import() # Instantiates

       >> I(filenames=list_of_filenames,
            resolution=1e5,
            skiprows=6) #  Imports data

    To access data, use:

       >> mvt_data = I.Get('mvt_tagname_000') # Get memmap

       >> freq_data = I.Get('freq_tagname_000')

    """
    def __init__(self, **kwargs):
        self.Initialize(kwargs)

    def __call__(self, **kwargs):
        self.Initialize(kwargs)
        self.ImportData()

    def Initialize(self, kwargs):
        self.opt = {
            'filenames': list(), # list of filenames (with full path)
            'folder': '', # name of folder (full path)
            'input_format': 'txt', # not necessary unless all files are of same type
            'store_format': 'ndarray', # "
            'output_format': 'npy', # "
            'skiprows': 2, # Skips rows for np.loadtxt()
            'resolution': 5e3 # Set import resolution
        }
        self._NeuralList = list() # Create a new list of neurons
        self._MvtData = dict() # Dict that stores mvt data arrays as (Name: array)
        self._FreqData = dict() # Dict that stores frequency data arrays as (Name: array)
        self.LoadDict = LoadDict() # Instance of LoadDict class
        self.opt.update(kwargs) # Update options from keywords
        self._SetVars() # Store these options as class attributes

    def _SetVars(self):
        # Stores keyword options as class attributes
        for k, v in self.opt.items():
            setattr(self, k, v)

    def SetParams(self, param_dict):
        """ Update parameters after instantiation
        """
        self.opt.update(param_dict)
        self._SetVars()

    def ImportData(self, fns=None):
        """ Main Data import function """
        # Check if given list of filenames or a folder
        if self.IsFilenames(): filenames = self.filenames
        elif fns is not None: filenames = fns
        elif self.IsFolder() and not self.IsFilenames() and fns is None: 
            filenames = list(map(lambda fff: os.path.join(self.folder, fff), 
                                        os.listdir(self.folder)))
        if len(filenames) > 0:
            # Iterate through filenames, and store correct import function and filename
            self._Data = list(map(lambda fn: self._ConsultLoadDict(fn), filenames))
            for fimp, fna in self._Data:
                getattr(self, fimp)(fna) # Perform import function for each file
            if len(self._NeuralList) > 0:
                self._ToFreq() # If there is uncomputed raw neural data, compute it
                self._NeuralList = list() # Reset list of raw neural data

    def Get(self, tagname):
        """
            >> I = Import(filenames)
            >> mvt_data = I.Get("mvt_000") 
        :return: memmap corresponding to tagname
        """
        if tagname in self._MvtData.keys(): return self._MvtData[tagname]
        elif tagname in self._FreqData.keys(): return self._FreqData[tagname]

    def IsFolder(self):
        return self.folder != ''

    def IsFilenames(self):
        return len(self.filenames) != 0

    def _ConsultLoadDict(self, filename):
        # LoadDict maps file type/ext to correct import function
        filename0 = filename # Keep full path intact
        filename1 = filename.split(os.sep)[-1] # separate from path
        keys, import_function = self.LoadDict(filename) # Consult LoadDict
        # :returns: correct import function name, original filename
        return import_function, filename0

    def _MvtTxt2Np(self, filename):
        """ [Mvt###.txt] -> [np.ndarray] """
        return self.ImportMovementData(filename, fmt='txt')

    def _MvtNpy2Np(self, filename):
        """ [Mvt###.npy] -> [np.ndarray] """
        return self.ImportMovementData(filename, fmt='npy')

    def _NeuralTxt2Np(self, filename):
        """ [Neuron###.txt] -> [np.ndarray] 
            appends (filename, array) to _NeuralList list
            for later computation """
        self._NeuralList.append((filename, 
                                np.loadtxt(filename, skiprows=self.skiprows)))

    def _NeuralNpy2Np(self, filename):
        """ [Neuron###.npy] -> [np.ndarray] 
            appends (filename, array) to _NeuralList list
            for later computation """
        self._NeuralList.append((filename, np.load(filename)))

    def ImportMovementData(self, filename,
                           nr_trace=None,
                           fmt='txt'):
        """
            Used for importing movement data, 
            exports as .npy
            :return: np.array([time points, 1])
        """
        def iter_loadtxt(filename, delimiter='\t',
                        skiprows=6, dtype=np.float32):
            def iter_func():
                with open(filename, 'r') as infile:
                    for _ in range(skiprows):
                        next(infile)
                    for line in infile:
                        line = line.rstrip().split(delimiter)
                        for item in line:
                            yield dtype(item)
                iter_loadtxt.rowlength = len(line)
            data = np.fromiter(iter_func(), dtype=dtype)
            data = data.reshape((-1, iter_loadtxt.rowlength))
            return data
        if fmt == 'txt':
            try:
                wv = iter_loadtxt(filename, delimiter='\t', skiprows=6) # Load waveform from txt
            except ValueError:
                wv = iter_loadtxt(filename, delimiter='\t', skiprows=7) # Load waveform from txt
        elif fmt == 'npy': wv = np.load(filename) # Load waveform from npy
        if nr_trace is not None: wv = wv[:, nr_trace] # check if trace provided
        mvt_ntf = NamedTemporaryFile(prefix='mvt_', suffix='.npy', delete=False) # Temporary file
        mvt_mem = np.memmap(mvt_ntf.name, dtype=wv.dtype, shape=wv.shape) # Memmap
        mvt_mem[:] = wv[:] # Store in memmap
        tagname = ''.join(list(filter(lambda tl: tl.isalpha(), filename.split(os.sep)[-1].split('.')[0]))) # Tagname of data (use to find later)
        self._MvtData.update({tagname: mvt_mem}) # Store temp in _MvtData dict for quick access

    def _ToFreq(self):
        # Convert neural data to frequency response data
        # :return: np.array([time points, neurons])
        tagname = ''.join(list(filter(lambda tl: tl.isalpha(), self._NeuralList[0][0].split(os.sep)[-1].split('.')[0]))) # Get tagname
        data = list(map(lambda naa: naa[1], self._NeuralList)) # Get stored raw neural data
        nr_pts = self.resolution # Get output resolution
        def where_pts(nr_pts, time_space, delta, freq, datum):
            for ii in np.arange(nr_pts):
                ii = int(ii)
                # Count number of spike times that fall between t[i] and t[i + 1]
                count = len(datum[np.where((datum < time_space[ii + 1]) & (datum > time_space[ii]))])
                # The frequency here is the count divided by a single output timestep
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
            freq_ntf = NamedTemporaryFile(prefix='freq_', suffix='.npy', delete=False)
            # np.save(freq_ntf, freq) # Store frequency in temporary npy file
            freq_mem = np.memmap(freq_ntf.name, dtype=freq.dtype, shape=freq.shape) # Create a memmap for quick access
            freq_mem[:] = freq[:] # Store in memmap
            self._FreqData.update({tagname: freq_mem}) # Store memmap in _FreqData dict

class LoadDict(dict):

    """ Used by the Import class:
        [file type/ext] -> [correct import function]
    """

    def __init__(self):
        self.opt = {
            ('neuron000', 'txt'): '_NeuralTxt2Np',
            ('mvt', 'txt'): '_MvtTxt2Np',
            ('neuron000', 'npy'): '_NeuralNpy2Np',
            ('mvt', 'npy'): '_MvtNpy2Np'
        }
        super(dict).__init__(dict, self)

    def __call__(self, filename):
        return self.CompareName(filename)

    def CompareName(self, filename):
        # :returns: ('key1', 'key2'), 'import_function'
        if filename.split('.')[-1] == 'asc': return ('mvt', 'txt'), '_MvtTxt2Np'
        diffs = list(map(lambda kv: (kv, distance.nlevenshtein('.'.join(kv[0]), filename, method=2)), self.opt.items()))
        return min(diffs, key=lambda itt: itt[1])[0]

class Data:
    def __init__(self, **kwargs):
        pass

# if __name__ == '__main__':
#     import time
#     narr = [np.random.random((int(1e6),1)) for i in range(33)]
#     t0=time.time()
#     I = Import()
#     print I.ToFreq(narr, 1000).shape
#     print time.time() - t0