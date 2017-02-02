# -*- coding: utf-8 -*-
"""
Generates custom runcards for tunes variations
"""
__author__ = "Stefano Carrazza & Simone Alioli"
__version__= "1.0.0"

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor


class NNModel(object):

    def __init__(self, x, y, seed = 0):
        """Allocate random seed"""
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.model = None

    def build_model(self, optimizer='rmsprop', init='glorot_uniform'):
        """build neural network model"""
        model = Sequential()
        model.add(Dense(output_dim=6, input_dim=self.x.shape[1], init=init, activation='tanh'))
        model.add(Dense(output_dim=15, input_dim=6, init=init, activation='tanh'))
        model.add(Dense(output_dim=self.y.shape[1], input_dim=15, init=init, activation='linear'))
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def fit(self):
        """apply fitting proceduce"""
        print('Fitting...')
        self.model = self.build_model()
        h = self.model.fit(self.x, self.y, nb_epoch=1000, batch_size=20, verbose=0)
        print('Final loss function: %f' % h.history['loss'][-1])

    def predict(self, x):
        """compute prediction"""
        return self.model.predict(x)

    def save(self, folder):
        """save model to file"""
        self.model.save('%s/model.hd5' % folder)

    def load(self, file):
        """load model from file"""
        self.model = load_model(file)

