# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
"""
import numpy as np
import scipy
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

        # Compute degrees of freedom for chi2
        self.dof = self.runs.weighted_dof

    def chi2(self, x):
        """Reduced chi2 estimator (weighted, eventually)"""
        x = x.reshape(1,self.runs.x_scaled.shape[1])
        prediction = self.model.predict(x, scaled_x = True, scaled_y = False)
        return stats.chi2(prediction, self.truth, self.truth_error2, weights=self.runs.y_weight, dof=self.dof)

    def unweighted_chi2(self, x):
        """Reduced chi2 estimator (always unweighted)"""
        x = x.reshape(1,self.runs.x_scaled.shape[1])
        prediction = self.model.predict(x, scaled_x = True, scaled_y = False)
        return stats.chi2(prediction, self.truth, self.truth_error2, dof=self.runs.unweighted_dof)

    @abstractmethod
    def minimize(self):
        pass

    def compute_errors(self, best_x_scaled):
        """"""

        N = 1000 # points
        errors = []
        best_x_unscaled = self.runs.unscale_x(best_x_scaled)
        delta_chi2 = scipy.stats.chi2(self.dof).ppf(0.682689492137)  / self.dof

        # Iterate over all params
        for axis in range(self.runs.x_scaled.shape[1]):

            # Scan the x-space and compute the chi2 profile
            x_scan = np.linspace(np.min(self.runs.x_scaled[:, axis]),
                                 np.max(self.runs.x_scaled[:, axis]),
                                 N)
            chi2_values = np.zeros(N)
            x_unscaled  = np.zeros(N)
            for index, param in enumerate(x_scan):
                a = np.array(best_x_scaled)
                a[axis] = param
                chi2_values[index] = self.chi2(a)
                x_unscaled[index]  = self.runs.unscale_x(a)[axis]

            # Compute where chi2 - min(chi2) ~= delta_chi2
            intersec_idxs = np.argwhere(np.diff(np.sign(
                                chi2_values - np.min(chi2_values) - delta_chi2
                            )))

            # Find the intersection points which are the closest to the minimum
            x_intersections = x_unscaled[intersec_idxs]
            lower_values  = [best_x_unscaled[axis]-item for item in x_intersections if item < best_x_unscaled[axis]]
            higher_values = [item-best_x_unscaled[axis] for item in x_intersections if item > best_x_unscaled[axis]]
            if len(lower_values)==0:
                lower_values.append(np.zeros(1))
            if len(higher_values)==0:
                higher_values.append(np.zeros(1))
            errors.append([min(lower_values), min(higher_values)])

        return errors

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
        best_std = self.compute_errors(self.result[0])

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
                best_x_scaled = np.array(layer.get_weights()).reshape(-1)
                best_x = self.runs.unscale_x(best_x_scaled)
        best_std = self.compute_errors(best_x_scaled)

        return best_x, best_std

    def chi2_Keras_loss(self, y_true, y_pred):
        """Custom chi2 loss function ready to be plugged in in Keras"""
        return stats.chi2_tf(y_true, y_pred, self.truth_error2 / (self.runs.y_std ** 2), weights=self.runs.y_weight, dof=self.dof)
