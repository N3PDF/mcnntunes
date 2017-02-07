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


class NNModel(object):

    def __init__(self, seed = 0):
        """Allocate random seed"""
        np.random.seed(seed)
        self.input_dim = 0
        self.output_dim = 0
        self.model = None

    def build_model(self, optimizer='rmsprop', loss='mse', architecture=None, actfunction=None, init='glorot_uniform'):
        """build neural network model"""
        model = Sequential()

        nsizes = [self.input_dim] + [l for l in architecture] + [self.output_dim]
        actfun = [actfunction for l in range(len(architecture))] + ['linear']

        for l in range(len(nsizes)-1):
            model.add(Dense(output_dim=nsizes[l+1], input_dim=nsizes[l], init=init, activation=actfun[l]))

        model.compile(loss=loss, optimizer=optimizer)
        return model

    def fit(self, x, y, setup):
        """apply fitting procedure"""
        show('\n- Using no scan setup:')
        for key in setup.keys():
            show('  - %s : %s' % (key, setup.get(key)))

        self.input_dim = x.shape[1]
        self.output_dim = y.shape[1]
        self.model = self.build_model(setup['optimizer'], setup['loss'],
                                      setup['architecture'], setup['actfunction'])
        h = self.model.fit(x, y, nb_epoch=setup['nb_epoch'], batch_size=setup['batch_size'], verbose=0)
        show('\n- Final loss function: %f' % h.history['loss'][-1])
        self.loss = h.history['loss']

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

