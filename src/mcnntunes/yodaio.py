# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
"""
import yoda, pickle
import numpy as np
from mcnntunes.tools import error, show, info


class Data(object):

    annotation_tag = 'Tune_Parameter'

    def __init__(self, filenames, patterns, unpatterns, weightrules, expData=False):
        """the data container"""
        self.expData = expData

        show('\n- Reading yoda files...')
        yoda_histograms = []
        for filename in filenames:
            r = yoda.read(filename, patterns=patterns, unpatterns=unpatterns)
            if len(r) == 0 and expData == False:
                error('Empty histograms following pattern %s' % patterns)
            elif len(r) == 0:
                info('Empty histograms in %s for pattern %s' % (filename, patterns))
            yoda_histograms.append(r)

        if expData:
            tmp_yoda_histograms = dict(pair for d in yoda_histograms for pair in d.items())
            yoda_histograms = []
            yoda_histograms.append({})
            for key in tmp_yoda_histograms:
                yoda_histograms[0][key[4:]] = tmp_yoda_histograms[key]

        entries = len(yoda_histograms)
        if entries == 0:
            error('Problem with input histograms')

        # scan space using first yoda file
        input_param = []
        output_size = 0
        obj = yoda_histograms[0]
        for key in sorted(obj):
            try:
                h = obj.get(key)
            except:
                error('Cannot open histogram %s' % key)
            output_size += len(h.points())
            if not expData:
                param = [item for item in h.annotations() if self.annotation_tag in item]
                for p in param:
                    if p not in input_param:
                        input_param.append(p)
            show('  ==] Loaded %s from %d runs (dof=%d)' % (key, entries, len(h.points())))

        if not expData:
            input_size = len(input_param)
            show('\n- Detected input parameters: %d' % input_size)
            for item in input_param:
                show('  ==] %s' % item)
            self.params = input_param
            self.x = np.zeros(shape=(entries, input_size))

        show('\n- Detected output bins: %d' % output_size)

        self.y = np.zeros(shape=(entries, output_size))
        self.yerr = np.zeros(shape=(entries, output_size))
        self.y_weight = np.zeros(output_size) #array of weights

        # load data from files
        self.plotinfo = []
        for i, file in enumerate(yoda_histograms):
            index = 0
            for key in sorted(file):
                h = file.get(key)
                if not expData:
                    for j, inparam in enumerate(input_param):
                        self.x[i, j] = h.annotation(inparam)
                data_x = np.zeros(len(h.points()))
                data_xerrm = np.zeros(len(h.points()))
                data_xerrp = np.zeros(len(h.points()))
                data_y = np.zeros(len(h.points()))
                data_yerr = np.zeros(len(h.points()))
                data_weight = np.zeros(len(h.points()))
                for t, p in enumerate(h.points()):
                    self.y[i,index] = p.y()
                    self.yerr[i,index] = p.yErrAvg()
                    if i==0: # Need just one run
                        self.y_weight[index] = self.get_weight(key,weightrules,t+1,p.x(), verbose=True)
                    index += 1
                    data_x[t] = p.x()
                    data_xerrm[t] = p.xErrs()[0]
                    data_xerrp[t] = p.xErrs()[1]
                    data_y[t] = p.y()
                    data_yerr[t] = p.yErrAvg()
                    data_weight[t] = self.get_weight(key,weightrules,t+1,p.x())
                    if p.y() == 0:
                        info('Histogram %s has empty entries' % key)
                self.plotinfo.append({'title': key.replace('/REF',''),
                                      'x': data_x,
                                      'y': data_y,
                                      'yerr': data_yerr,
                                      'xerr-': data_xerrm,
                                      'xerr+': data_xerrp,
                                      'weight': data_weight})
        show('\n- Data loaded successfully')

        # Calculate dof
        if not expData:
            self.unweighted_dof = self.y.shape[1] - len(self.params)
            if self.unweighted_dof <= 0:
                error(f'Error: invalid dof={self.unweighted_dof}.')
            self.weighted_dof = np.sum(self.y_weight) - len(self.params)
            if self.weighted_dof <= 0:
                error(f'Error: invalid weighted dof={self.weighted_dof}.')

        if not expData:
            self.x_mean = np.mean(self.x, axis=0)
            self.x_std  = np.std(self.x, axis=0)
            self.x_scaled = (self.x - self.x_mean) / self.x_std
            self.y_mean = np.mean(self.y, axis=0)
            self.y_std = np.std(self.y, axis=0)
            self.y_scaled = (self.y - self.y_mean) / self.y_std
        else:
            self.y_mean = np.mean(self.y, axis=1)
            self.y_scaled = self.y-self.y_mean

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

    def get_weight(self, pattern, weightrules, bin, x_bin, verbose=False):
        """Check all weight rules to see if one corresponds to
        that bin. If not, sets the weight to 1."""
        weight = 1 # default weight
        for rule in weightrules:
            if rule['pattern'] == pattern:

                # In case of bin index
                if rule['condition_type'] == 'bin_index':
                    if rule['bin_index'] == bin:
                        weight = rule['weight']
                        if verbose:
                            show('  ==] Set weight of bin %d of histogram %s to %0.2f' % (bin, pattern, weight))

                # In case of interval
                elif rule['condition_type'] == 'interval':
                    if (rule['left_endpoint'] == '-inf') and (rule['right_endpoint'] == '+inf'):
                        weight = rule['weight']
                        if verbose:
                            show('  ==] Set weight of bin %d of histogram %s to %.2f' % (bin, pattern, weight))
                    elif (rule['left_endpoint'] == '-inf') and (rule['right_endpoint'] == '-inf'):
                        error('Error: Potentially unwanted [-inf,-inf] condition found in a weightrule.')
                    elif (rule['left_endpoint'] == '+inf') and (rule['right_endpoint'] == '+inf'):
                        error('Error: Potentially unwanted [+inf,+inf] condition found in a weightrule.')
                    elif (rule['left_endpoint'] == '+inf') and (rule['right_endpoint'] == '-inf'):
                        return weight # do nothing
                    elif rule['left_endpoint'] == '+inf':
                        return weight # do nothing
                    elif rule['right_endpoint'] == '-inf':
                        return weight # do nothing
                    elif rule['left_endpoint'] == '-inf':
                        if x_bin <= rule['right_endpoint']:
                            weight = rule['weight']
                            if verbose:
                                show('  ==] Set weight of bin %d of histogram %s to %.2f' % (bin, pattern, weight))
                    elif rule['right_endpoint'] == '+inf':
                        if x_bin >= rule['left_endpoint']:
                            weight = rule['weight']
                            if verbose:
                                show('  ==] Set weight of bin %d of histogram %s to %.2f' % (bin, pattern, weight))
                    elif (x_bin >= rule['left_endpoint']) and (x_bin <= rule['right_endpoint']):
                            weight = rule['weight']
                            if verbose:
                                show('  ==] Set weight of bin %d of histogram %s to %.2f' % (bin, pattern, weight))

        return weight

    @classmethod
    def load(cls, stream):
        """read yaml from stream"""
        obj = pickle.load(open(stream,'rb'))
        show('\n- Loaded data from %s' % stream)
        if not obj.expData:
            show('\n- Detected %d inputs and %d outputs for %d entries' %(obj.x.shape[1], obj.y.shape[1], obj.x.shape[0]))
        return obj
