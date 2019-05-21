# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

from abc import ABC, abstractmethod
import pickle, h5py
import numpy as np
from .tools import show, error, make_dir
from keras.models import Sequential, load_model
from keras.layers.core import Dense
import matplotlib.pyplot as plt


def build_model(input_dim=None, output_dim=1,
                optimizer='rmsprop', loss='mse',
                architecture=None, actfunction=None,
                init='glorot_uniform'):
    """build neural network model"""
    model = Sequential()

    nsizes = [input_dim] + [l for l in architecture] + [output_dim]
    actfun = [actfunction for l in range(len(architecture))] + ['linear']

    for l in range(len(nsizes) - 1):
        model.add(Dense(units=nsizes[l + 1], input_dim=nsizes[l], kernel_initializer=init, activation=actfun[l]))

    model.compile(loss=loss, optimizer=optimizer)
    return model

class Model(ABC):
    """Abstract class for a generic model"""

    def __init__(self, runs, seed = 0):
        """Store data attributes"""
        self.runs = runs
        self.seed = seed
        self.READY = False

    @abstractmethod
    def build_and_train_model(self, setup):
        pass

    @abstractmethod
    def save_model_and_plots(self, output_path):
        pass

    @abstractmethod
    def search_and_load_model(self, input_path):
        pass

    @abstractmethod
    def predict(self, x, scaled_x = True, scaled_y = True):
        pass

class DirectModel(Model):
    """This model predicts the MC run output giving the input parameters."""

    def __init__(self, runs, seed = 0):
        """Set data attributes"""
        Model.__init__(self, runs, seed = 0)
        self.model_type = 'DirectModel'
    
    def build_and_train_model(self, setup):
        """Build and train n_bins FullyConnected models"""
        
        self.per_bin_nns = []
        for bin in range(1, self.runs.y.shape[1]+1):
            nn = PerBinModel(self.seed)
            show(f'\n- Fitting bin {bin}')
            nn.fit(self.runs.x_scaled, self.runs.y_scaled[:,bin-1], setup)
            self.per_bin_nns.append(nn)

        # Update READY flag
        self.READY = True

    def save_model_and_plots(self, output_path):
        """Save model, losses and plots in the output path"""
        if not self.READY:
            error('Error: nothing to save, call build_and_train_model or search_and_load_model first.')
 
        make_dir(f'{output_path}')

        for bin in range(1, self.runs.y.shape[1]+1):
            make_dir(f'{output_path}/model_bin_{bin}')
            save(self.per_bin_nns[bin-1].model, self.per_bin_nns[bin-1].loss, f'{output_path}/model_bin_{bin}/model.h5')
            self.per_bin_nns[bin-1].plot(f'{output_path}/model_bin_{bin}', self.runs.x_scaled, self.runs.y_scaled[:,bin-1])        

    def search_and_load_model(self, input_path):
        """Search for models in the input path (and load them)."""
        if self.READY:
            error('Error: loading model into a ready-to-predict model object.')

        self.per_bin_nns = []
        for bin in range(1, self.runs.y.shape[1]+1):
            nn = PerBinModel()
            nn.model, nn.loss = load(f'{input_path}/model_bin_{bin}/model.h5')
            nn.model.name = f'bin_predictor_{bin}'
            self.per_bin_nns.append(nn)

        # Update READY flag
        self.READY = True

    def predict(self, x, scaled_x = True, scaled_y = True):
        """Predicts the y passing the x"""

        if not self.READY:
            error('Error: trying to predict with an untrained model.')

        if not scaled_x:
            x = self.runs.scale_x(x)

        prediction = np.array([nn.predict(x) for nn in self.per_bin_nns]).reshape(-1)

        if not scaled_y:
            prediction = self.runs.unscale_y(prediction)

        return prediction

class PerBinModel(object):

    def __init__(self, seed = 0):
        """Allocate random seed"""
        np.random.seed(seed)
        self.input_dim = 0
        self.output_dim = 0
        self.model = None

    def fit(self, x, y, setup):
        """apply fitting procedure"""
        show('\n- Using no scan setup:')
        for key in setup.keys():
            show('  - %s : %s' % (key, setup.get(key)))

        self.model = build_model(x.shape[1], 1,
                                 setup['optimizer'], 'mse',
                                 setup['architecture'], setup['actfunction'])

        h = self.model.fit(x, y, epochs=setup['nb_epoch'], batch_size=setup['batch_size'], verbose=0)
        self.loss = h.history['loss'] + [self.model.evaluate(x,y,verbose=0)]
        show('\n- Final loss function: %f' % self.loss[-1])

    def predict(self, x):
        """compute prediction"""
        return self.model.predict(x)

    def plot(self, path, x, y):
        """"""
        for i in range(x.shape[1]):
            plt.figure()
            plt.plot(x[:,i], y, 'o', color='c', label='MC run')
            plt.plot(x[:,i], self.predict(x), 'o', color='r', alpha=0.5, label='NN model')
            plt.xlabel('x%d' % i)
            plt.ylabel('bin value')
            plt.legend(loc='best')
            plt.grid()
            plt.savefig('%s/x%d.svg' % (path, i))
            plt.close()

        plot_losses(path, self.loss)

class InverseModel(Model):
    """ This model predicts the input parameters giving the MC run output"""

    def __init__(self, runs, seed = 0):
        """Set data attributes"""
        Model.__init__(self, runs, seed = 0)
        self.model_type = 'InverseModel'

    def build_and_train_model(self, setup):
        """Build and train a FullyConnected model"""

        # Allocate random seed
        np.random.seed(self.seed)

        # Print setup
        show('\n- Setup:')
        for key in setup.keys():
            show('  - %s : %s' % (key, setup.get(key)))

        # Rename inputs and outputs for readability
        x = self.runs.y_scaled
        y = self.runs.x_scaled

        # Build and train the model
        self.model = build_model(x.shape[1], y.shape[1], setup['optimizer'], 'mse', setup['architecture'], setup['actfunction'])
        h = self.model.fit(x, y, epochs=setup['nb_epoch'], batch_size=setup['batch_size'], verbose=0)
        self.loss = h.history['loss'] + [self.model.evaluate(x,y,verbose=0)]
        show('\n- Final loss function: %f' % self.loss[-1])


        # Update READY flag
        self.READY = True

    def save_model_and_plots(self, output_path):
        """Save model, losses and plots in the output path"""
        if not self.READY:
            error('Error: nothing to save, call build_and_train_model or search_and_load_model first.')
        
        make_dir(f'{output_path}')

        # Save the losses and the model
        save(self.model, self.loss, f'{output_path}/model.h5')
        plot_losses(output_path, self.loss)

    def search_and_load_model(self, input_path):
        """Search for a model in the input path (and load it)."""
        if self.READY:
            error('Error: loading model into an already complete model object.')

        self.model, self.loss = load(f'{input_path}/model.h5')

        # Update READY flag
        self.READY = True

    def predict(self, x, scaled_x = True, scaled_y = True):
        """Predicts the y passing the x. Notice that the notation is
        the opposite of the one in the Data class"""

        if not self.READY:
            error('Error: trying to predict with an untrained model.')

        if not scaled_x:
            x = self.runs.scale_y(x)

        prediction = self.model.predict(x).reshape(-1)

        if not scaled_y:
            prediction = self.runs.unscale_x(prediction)

        return prediction

def get_model(model_type, runs, seed = 0):
    """Return a Model object, discriminating between different model type"""
    if model_type == 'DirectModel':
        return DirectModel(runs, seed)
    elif model_type == 'InverseModel':
        return InverseModel(runs, seed)
    else:
        error('Error: invalid model type.')

def save(model, loss, file):
        """save model to file"""
        model.save(file)
        with h5py.File(file, 'r+') as f:
            del f['optimizer_weights']
            f.close()

        pickle.dump(loss, open(f'{file}.p', 'wb'))
        show(f'\n- Model saved in {file}')

def load(file):
        """load model from file"""
        model = load_model(file)
        loss = pickle.load(open(f'{file}.p', 'rb'))
        show(f'\n- Model loaded from {file}')

        return model, loss

def plot_losses(path, training_loss, validation_loss = None):
    """"""
    plt.figure()
    plt.plot(training_loss, label='Training loss')
    if not (validation_loss is None):
        plt.plot(validation_loss, label='Validation loss')
    plt.title('loss function')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.savefig(f'{path}/loss.svg')
    plt.close()