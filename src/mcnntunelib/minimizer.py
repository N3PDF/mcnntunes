# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import cma


class CMAES(object):

    def __init__(self, model, truth, runs, useBounds = True, output='.'):
        """"""
        self.model = model
        self.truth = truth.y[0]
        self.truth_error = truth.yerr[0]
        self.runs = runs

        s0max = np.max(runs.x_scaled, axis=0).tolist()
        s0min = np.min(runs.x_scaled, axis=0).tolist()
        center = 0
        sigma = 1
        if useBounds:
            opts = {'bounds': [s0min, s0max], 'verb_filenameprefix': '%s/cma-' % output }
        else:
            opts = {'verb_filenameprefix': '%s/cma-' % output }

        print('\n- Minimizer setup:')
        if useBounds: print('  - bounds: on')
        else: print('  - bounds: off')
        print('  - centers: %f' % center)
        print('  - sigma: %f ' % sigma)

        self.es = cma.CMAEvolutionStrategy(self.model.model.input_shape[1] * [center], sigma, opts)

    def chi2(self, x):
        model = self.runs.unscale_y(self.model.predict(x.reshape(1,self.runs.x_scaled.shape[1])).reshape(len(self.truth)))
        return np.mean(np.square((model-self.truth)/self.truth_error))

    def minimize(self):
        """"""
        self.es.optimize(self.chi2)
        return self.es.result()