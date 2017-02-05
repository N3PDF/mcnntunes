# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import yoda, pickle
import numpy as np
from tools import error, show


class Data(object):

    annotation_tag = 'Tune_Parameter'

    def __init__(self, filenames, patterns, unpatterns, expData=False):
        """the data container"""
        show('\n- Reading yoda files...')
        yoda_histograms = []
        for i, filename in enumerate(filenames):
            r = yoda.read(filename, patterns=patterns, unpatterns=unpatterns)
            if len(r) == 0 and expData == False:
                error('Empty histograms following pattern %s' % patterns)
            elif len(r) == 0:
                show('Warning: empty histograms in %s for pattern %s' % (filename, patterns))
            yoda_histograms.append(r)

        entries = len(yoda_histograms)
        if entries == 0:
            error('Problem with input histograms')

        for keys in yoda_histograms[0]:
            show('  ==] Loaded %s from %d runs' % (keys, entries))

        # scan space using first yoda file
        input_param = []
        output_size = 0
        obj = yoda_histograms[0]
        for key in obj:
            try:
                h = obj.get(key)
            except:
                error('Cannot open histogram %s' % key)
            output_size += len(h.points)
            if not expData:
                param = [item for item in h.annotations if self.annotation_tag in item]
                for p in param:
                    if p not in input_param:
                        input_param.append(p)

        if not expData:
            input_size = len(input_param)
            show('\n- Detected input parameters: %d' % input_size)
            for item in input_param:
                show('  ==] %s' % item)
            self.params = input_param
            self.x = np.zeros(shape=(entries, input_size))

        show('\n- Detected output parameters: %d' % output_size)

        self.y = np.zeros(shape=(entries, output_size))
        self.yerr = np.zeros(shape=(entries, output_size))

        # load data from files
        try:
            for i, file in enumerate(yoda_histograms):
                index = 0
                for key in file:
                    h = file.get(key)
                    if not expData:
                        for j, inparam in enumerate(input_param):
                            self.x[i, j] = h.annotation(inparam)
                    for p in h.points:
                        self.y[i,index] = p.y
                        self.yerr[i,index] = p.yErrAvg
                        index += 1
        except:
            error("Error: yoda files are not consistent.")

        show('\n- Data loaded successfully')

        if not expData:
            self.x_mean = np.mean(self.x, axis=0)
            self.x_std  = np.std(self.x, axis=0)
            self.x_scaled = (self.x - self.x_mean) / self.x_std

        self.y_mean = np.mean(self.y, axis=0)
        self.y_std  = np.std(self.y, axis=0)
        self.y_scaled = (self.y - self.y_mean) / self.y_std

        show('\n- Data standardized successfully')

    def unscale_x(self, array):
        """"""
        return array*self.x_std + self.x_mean

    def unscale_y(self, array):
        """unstandardize y-prediction"""
        return array*self.y_std + self.y_mean

    def scale_x(self, array):
        """"""
        return (array - self.x_mean) / self.x_std

    def scale_y(self, array):
        """"""
        return (array - self.y_mean) / self.y_std

    def save(self, path):
        """"""
        pickle.dump(self, open(path, 'wb'))
        show('\n- Data saved in %s' % path)

    @classmethod
    def load(cls, stream):
        """read yaml from stream"""
        obj = pickle.load(open(stream,'rb'))
        show('\n- Loaded data from %s' % stream)
        show('\n- Detected %d inputs and %d outputs for %d entries' %(obj.x.shape[1], obj.y.shape[1], obj.x.shape[0]))
        return obj
