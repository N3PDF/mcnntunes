# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import cma
from cma import fmin


class CMAES(object):

    def __init__(self, models, truth, runs, useBounds = True, output='.'):
        """"""
        self.models = models
        self.truth = truth.y[0]
        self.truth_error = truth.yerr[0]
        self.runs = runs
        self.output = output
        s0max = np.max(runs.x_scaled, axis=0).tolist()
        s0min = np.min(runs.x_scaled, axis=0).tolist()
        self.center = 0
        self.sigma = 1
        if useBounds:
            self.opts = {'bounds': [s0min, s0max], 'verb_filenameprefix': '%s/cma-' % output }
        else:
            self.opts = {'verb_filenameprefix': '%s/cma-' % output }

        print('\n- Minimizer setup:')
        if useBounds: print('  - bounds: on')
        else: print('  - bounds: off')
        print('  - centers: %f' % self.center)
        print('  - sigma: %f ' % self.sigma)

        #self.es = cma.CMAEvolutionStrategy(center, sigma, opts)

    def chi2(self, x):
        prediction = np.zeros(self.truth.shape[0])
        for i, model in enumerate(self.models):
            prediction[i] = model.predict(x.reshape(1,self.runs.x_scaled.shape[1]))
        prediction = self.runs.unscale_y(prediction)
        return np.mean(np.square((prediction-self.truth)/self.truth_error))

    def minimize(self):
        """"""
        res = fmin(self.chi2,
                   str([self.center] * self.runs.x_scaled.shape[1]),
                   self.sigma,
                   self.opts,
                   restarts=3,
                   bipop=True)
        return res
        #self.es.optimize(self.chi2)
        #return self.es.result()