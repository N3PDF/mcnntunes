# -*- coding: utf-8 -*-
"""
Generates custom runcards for tunes variations
"""
__author__ = "Stefano Carrazza & Simone Alioli"
__version__= "1.0.0"

import yoda
import numpy as np


class Data(object):

    annotation_tag = 'Tune_Parameter'

    def __init__(self, filenames, pattern):
        """the data container"""

        print('Reading yoda files...')
        self.yodafiles = []
        for i, filename in enumerate(filenames):
            r = yoda.read(filename, patterns=pattern)
            if len(r) == 0:
                raise Exception('Empty histograms following pattern %s' % pattern)
            self.yodafiles.append(r)

        for keys in self.yodafiles[0]:
            print('Using %s from %d runs' % (keys, len(self.yodafiles)))

        self.trials = len(self.yodafiles)

    def scan_space(self):
        """extracting input and output size from first file"""
        try:
            self.input_param = []
            output_size = 0
            obj = self.yodafiles[0]
            for key in obj:
                h = obj.get(key)
                output_size += len(h.points)
                param = [item for item in h.annotations if self.annotation_tag in item]
                for p in param:
                    if p not in self.input_param:
                        self.input_param.append(p)
            input_size = len(self.input_param)
            print('Input parameters: %d' % input_size)
            print('Unique output parameters: %d' % output_size)

        except:
            raise Exception('Cannot find Tune_* parameters')

        self.x = np.zeros(shape=(self.trials, input_size))
        self.y = np.zeros(shape=(self.trials, output_size))
        self.load_data()

    def load_data(self):
        """load data from files"""
        try:
            for i, file in enumerate(self.yodafiles):
                index = 0
                for key in file:
                    h = file.get(key)
                    for j, inparam in enumerate(self.input_param):
                        self.x[i,j] = h.annotation(inparam)
                    for p in h.points:
                        self.y[i,index] = p.y
                        index += 1
        except:
            raise Exception("Yoda files are not consistent.")

    def standardize(self):
        """standardize data"""
        self.x_mean = np.mean(self.x, axis=0)
        self.x_std  = np.std(self.x, axis=0)
        self.y_mean = np.mean(self.y, axis=0)
        self.y_std  = np.std(self.y, axis=0)
        self.x_scaled = (self.x - self.x_mean) / self.x_std
        self.y_scaled = (self.y - self.y_mean) / self.y_std

    def unscale_y(self, array):
        """unstandardize y-prediction"""
        return array*self.y_std+self.y_mean