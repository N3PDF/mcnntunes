# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from cma import fmin
import mcnntunelib.stats as stats


class CMAES(object):

    def __init__(self, models, truth, runs, useBounds = True, output='.', truth_index=0):
        """"""
        self.models = models
        self.truth = truth.y[truth_index]
        self.truth_error2 = np.square(truth.yerr[truth_index]) + np.square(np.mean(runs.yerr, axis=0))
        self.runs = runs
        self.output = output
        s0max = np.max(runs.x_scaled, axis=0).tolist()
        s0min = np.min(runs.x_scaled, axis=0).tolist()
        self.center = 0
        self.sigma = 0.1

        self.opts = {'verb_filenameprefix': '%s/cma-' % output, 'tolfunhist': 0.01}
        if useBounds:
            self.opts['bounds'] = [s0min, s0max]

        print('\n- Minimizer setup:')
        if useBounds:
            print('  - bounds: on')
        else:
            print('  - bounds: off')
        print('  - centers: %f' % self.center)
        print('  - sigma: %f ' % self.sigma)

    def chi2(self, x):
        prediction = np.zeros(self.truth.shape[0])
        X = x.reshape(1,self.runs.x_scaled.shape[1])
        for i, model in enumerate(self.models):
            prediction[i] = model.predict(X)
        prediction = self.runs.unscale_y(prediction)
        return stats.chi2(prediction, self.truth, self.truth_error2, weights=self.runs.y_weight)

    def unweighted_chi2(self, x):
        prediction = np.zeros(self.truth.shape[0])
        X = x.reshape(1,self.runs.x_scaled.shape[1])
        for i, model in enumerate(self.models):
            prediction[i] = model.predict(X)
        prediction = self.runs.unscale_y(prediction)
        return stats.chi2(prediction, self.truth, self.truth_error2)

    def minimize(self, restarts):
        """"""
        print('  - restarts: %d ' % restarts)
        res = fmin(self.chi2,
                   str([self.center] * self.runs.x_scaled.shape[1]),
                   self.sigma,
                   self.opts,
                   restarts=restarts,
                   bipop=True)
        return res
