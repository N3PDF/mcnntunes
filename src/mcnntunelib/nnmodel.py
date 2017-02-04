# -*- coding: utf-8 -*-
"""
Generates custom runcards for tunes variations
"""
__author__ = "Stefano Carrazza & Simone Alioli"
__version__= "1.0.0"

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

    def build_model(self, optimizer='rmsprop', loss='mse', init='glorot_uniform'):
        """build neural network model"""
        model = Sequential()
        model.add(Dense(output_dim=6, input_dim=self.input_dim, init=init, activation='tanh'))
        model.add(Dense(output_dim=15, input_dim=6, init=init, activation='tanh'))
        model.add(Dense(output_dim=self.output_dim, input_dim=15, init=init, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def fit(self, x, y, setup):
        """apply fitting procedure"""
        show('\n- Using no scan setup:')
        for key in setup.keys():
            show('  - %s : %s' % (key, setup.get(key)))

        self.input_dim = x.shape[1]
        self.output_dim = y.shape[1]
        self.model = self.build_model(setup['optimizer'], setup['loss'])
        h = self.model.fit(x, y, nb_epoch=setup['nb_epoch'], batch_size=setup['batch_size'], verbose=0)
        show('\n- Final loss function: %f' % h.history['loss'][-1])

    def predict(self, x):
        """compute prediction"""
        return self.model.predict(x)

    def save(self, file):
        """save model to file"""
        self.model.save(file)

    def load(self, file):
        """load model from file"""
        self.model = load_model(file)
        show('\n- Model loaded from %s' % file)

