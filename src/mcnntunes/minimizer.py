# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from abc import ABC, abstractmethod
import pickle, h5py
from cma.evolution_strategy import fmin
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import mcnntunes.stats as stats
from mcnntunes.tools import show, error, make_dir


class Minimizer(ABC):
    """Abstract class for minimizing the chi2"""

    def __init__(self, runs, truth, model, output = None, truth_index = 0):
        """Set data attributes"""
        self.model = model
        self.truth = truth.y[truth_index]
        self.truth_error2 = np.square(truth.yerr[truth_index]) + np.square(np.mean(runs.yerr, axis=0))
        self.runs = runs
        if output is None:
            self.write_on_disk = False
        else:
            self.write_on_disk = True
            self.output = output

    def chi2(self, x):
        """Reduced chi2 estimator (weighted, eventually)"""
        x = x.reshape(1,self.runs.x_scaled.shape[1])
        prediction = self.model.predict(x, scaled_x = True, scaled_y = False)
        return stats.chi2(prediction, self.truth, self.truth_error2, weights=self.runs.y_weight)

    def unweighted_chi2(self, x):
        """Reduced chi2 estimator (always unweighted)"""
        x = x.reshape(1,self.runs.x_scaled.shape[1])
        prediction = self.model.predict(x, scaled_x = True, scaled_y = False)
        return stats.chi2(prediction, self.truth, self.truth_error2)

    @abstractmethod
    def minimize(self):
        pass

class CMAES(Minimizer):
    """Minimize the chi2 using CMA-EvolutionStrategy"""

    def __init__(self, runs, truth, model, output =  None, truth_index = 0, useBounds = True, restarts = 2):
        """Set data attributes"""
        Minimizer.__init__(self, runs, truth, model, output = output, truth_index = truth_index)

        s0max = np.max(runs.x_scaled, axis=0).tolist()
        s0min = np.min(runs.x_scaled, axis=0).tolist()
        self.center = 0
        self.sigma = 0.1
        self.restarts = restarts

        self.opts = {'tolfunhist': 0.01}
        if self.write_on_disk:
            self.opts['verb_filenameprefix'] = f'{self.output}/cma-'
        else:
            self.opts['verbose'] = -9
        if useBounds:
            self.opts['bounds'] = [s0min, s0max]

        show('\n- Minimizer setup:')
        if useBounds:
            show('  - bounds: on')
        else:
            show('  - bounds: off')
        show('  - centers: %f' % self.center)
        show('  - sigma: %f ' % self.sigma)
        show('  - restarts: %d ' % restarts)

    def minimize(self):
        """Minimize the chi2 and return the results"""
        self.result = fmin(self.chi2,
                   [self.center] * self.runs.x_scaled.shape[1],
                   self.sigma,
                   self.opts,
                   restarts=self.restarts,
                   bipop=True)

        # Unscale best_x and best_std
        best_x = self.runs.unscale_x(self.result[0])
        best_std = self.result[6] * self.runs.x_std

        return best_x, best_std

    def get_fmin_output(self):
        """"""
        try:
            return self.result
        except:
            error('Error: call to get_fmin_output without calling minimize')

class GradientMinimizer(Minimizer):
    """(experimental)"""

    def __init__(self, runs, truth, model, output = None, truth_index = 0):
        """Setting data attributes"""
        Minimizer.__init__(self, runs, truth, model, output = output, truth_index = truth_index)

    def minimize(self):
        """Build and train the minimizer, and return the results"""

        # Build the model
        input_tensor = Input(shape=(1,), name='input_layer')
        first_layer = Dense(self.runs.x.shape[1], activation='linear', name='parameters_layer', use_bias = False)(input_tensor)
        predictions = [nn.model(first_layer) for nn in self.model.per_bin_nns]
        predictor = Model(inputs=input_tensor, outputs=predictions)
        for i in range(len(predictor.layers)): # freeze everything except the parameters layer
            if predictor.layers[i].name != 'parameters_layer':
                predictor.layers[i].trainable = False
        predictor.compile(optimizer='adam', loss=self.chi2_Keras_loss)

        # Converte the unscaled target array to a scaled list
        scaled_truth = self.runs.scale_y(self.truth)
        target = [np.array([scaled_truth[i]]) for i in range(scaled_truth.shape[0])]

        # Train the model
        predictor.fit(x=np.ones(shape=(1,1)), y=target, epochs=5000, verbose=0)

        # Get the best parameters
        for layer in predictor.layers:
            if layer.name == 'parameters_layer':
                best_x = self.runs.unscale_x(np.array(layer.get_weights()).reshape(-1))
        best_std = np.ones(best_x.shape) # ISSUE

        return best_x, best_std

    def chi2_Keras_loss(self, y_true, y_pred):
        """Custom chi2 loss function ready to be plugged in in Keras"""
        return stats.chi2_tf(y_true, y_pred, self.truth_error2 / (self.runs.y_std ** 2), weights=self.runs.y_weight)
