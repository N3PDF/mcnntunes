# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import pickle
import numpy as np
from tools import show
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


def build_model(input_dim=None, output_dim=None,
                optimizer='rmsprop', loss='mse',
                architecture=None, actfunction=None,
                init='glorot_uniform'):
    """build neural network model"""
    model = Sequential()

    nsizes = [input_dim] + [l for l in architecture] + [output_dim]
    actfun = [actfunction for l in range(len(architecture))] + ['linear']

    for l in range(len(nsizes) - 1):
        model.add(Dense(output_dim=nsizes[l + 1], input_dim=nsizes[l], init=init, activation=actfun[l]))

    model.compile(loss=loss, optimizer=optimizer)
    return model


class NNModel(object):

    def __init__(self, seed = 0):
        """Allocate random seed"""
        np.random.seed(seed)
        self.input_dim = 0
        self.output_dim = 0
        self.model = None
        self.use_scan = 0

    def fit_noscan(self, x, y, setup):
        """apply fitting procedure"""
        show('\n- Using no scan setup:')
        for key in setup.keys():
            show('  - %s : %s' % (key, setup.get(key)))

        self.model = build_model(x.shape[1], y.shape[1],
                                 setup['optimizer'], 'mse',
                                 setup['architecture'], setup['actfunction'])

        h = self.model.fit(x, y, nb_epoch=setup['nb_epoch'], batch_size=setup['batch_size'], verbose=0)
        self.loss = h.history['loss'] + [self.model.evaluate(x,y,verbose=0)]
        show('\n- Final loss function: %f' % self.loss[-1])

    def fit_scan(self, x, y, setup, parallel):
        """"""
        self.use_scan = True
        show('\n- Using grid search scan setup:')
        for key in setup.keys():
            show('  - %s : %s' % (key, setup.get(key)))

        model = KerasRegressor(build_fn=build_model, verbose=0)
        param_grid = dict(setup)
        param_grid.pop('kfold')
        param_grid['input_dim'] = [x.shape[1]]
        param_grid['output_dim'] = [y.shape[1]]

        if parallel:
            njobs=-1
            show('\n- Using parallel mode')
        else: njobs = 1

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=njobs,
                            scoring='neg_mean_squared_error', cv=setup['kfold'],verbose=10)
        grid_result = grid.fit(x, y)

        # summary
        show("\n- Best: %f using %s" % (-grid_result.best_score_, grid_result.best_params_))

        means = -grid_result.cv_results_['mean_test_score']
        means2 = -grid_result.cv_results_['mean_train_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        print('\n- Grid Seach results:')
        for mean, stdev, mean2, param in zip(means, stds, means2, params):
            show("%f (%f) %f %f with: %r" % (mean, stdev, mean2, mean / mean2, param))

        self.model = grid_result.best_estimator_.model
        self.loss = [-grid_result.best_score_, self.model.evaluate(x,y,verbose=0)]

    def predict(self, x):
        """compute prediction"""
        return self.model.predict(x)

    def save(self, file):
        """save model to file"""
        self.model.save(file)
        pickle.dump(self.loss, open('%s.p' % file, 'wb'))

    def load(self, file):
        """load model from file"""
        self.model = load_model(file)
        self.loss = pickle.load(open('%s.p' % file, 'rb'))
        show('\n- Model loaded from %s' % file)
