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
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
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

        # Check and print setup
        fixed_setup = fix_setup_dictionary(setup)
        show('\n- Setup:')
        for key in fixed_setup.keys():
            if (key in fixed_setup["default_settings"] and key != 'data_augmentation'):
                show('  - %s : %s (default)' % (key, fixed_setup.get(key)))
            elif (key != 'default_settings' and key != 'data_augmentation'):
                show('  - %s : %s' % (key, fixed_setup.get(key)))
        if fixed_setup["data_augmentation"]:
            show("  Warning: data augmentation not available with this model.")

        self.model = build_model(x.shape[1], 1, get_optimizer(fixed_setup), 'mse', fixed_setup['architecture'],
                                    fixed_setup['actfunction'], fixed_setup["initializer"])
        h = self.model.fit(x, y, epochs=fixed_setup['epochs'], batch_size=fixed_setup['batch_size'], verbose=0)
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

        # Check and print setup
        fixed_setup = fix_setup_dictionary(setup)
        show('\n- Setup:')
        for key in fixed_setup.keys():
            if key in fixed_setup["default_settings"]:
                show('  - %s : %s (default)' % (key, fixed_setup.get(key)))
            elif key != 'default_settings':
                show('  - %s : %s' % (key, fixed_setup.get(key)))

        # Rename inputs and outputs for readability
        x = self.weight_mask(self.runs.y_scaled)
        y = self.runs.x_scaled
        x_err = self.weight_mask(self.runs.yerr / self.runs.y_std)

        # Build the model
        self.model = build_model(x.shape[1], y.shape[1], get_optimizer(fixed_setup), 'mse',
                                    fixed_setup['architecture'], fixed_setup['actfunction'], fixed_setup['initializer'])
        #Train the model
        if not fixed_setup['data_augmentation']: # standard procedure
            h = self.model.fit(x, y, epochs=fixed_setup['epochs'], batch_size=fixed_setup['batch_size'], verbose=0)
            self.loss = h.history['loss'] + [self.model.evaluate(x,y,verbose=0)]
        else: # data augmentation
            self.loss = []
            for i in range(fixed_setup['epochs']):
                h = self.model.fit(self.gaussian_noise(x, x_err), y, initial_epoch=i, epochs=i+1, batch_size=fixed_setup['batch_size'], verbose=0)
                self.loss.append(h.history['loss'][-1])
            self.loss.append(self.model.evaluate(x,y,verbose=0))

        # Print final loss function
        show('\n- Final loss function: %f' % self.loss[-1])


        # Update READY flag
        self.READY = True

    def gaussian_noise(self, x, x_err):
        """Add a gaussian noise with a different stddev for each bin of each run,
        according to the MC errors."""

        return x + x_err*np.random.normal(0., 1., x_err.shape)

    def weight_mask(self, x):
        """Removes the zero-weighted inputs from the dataset matrix.
        Use the convention rows=samples, cols=bins"""
        
        boolean_weight_mask = (self.runs.y_weight != 0)

        return x[:,boolean_weight_mask]

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

        prediction = self.model.predict(self.weight_mask(x)).reshape(-1)

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

def fix_setup_dictionary(setup):
    """
    This function checks if the setup dictionary has some missing keys
    (which will be replaced by a default value); moreover, it checks for
    unrecognised keys, which are probably typos.
    """

    fixed_setup = {}
    default_settings = []
    try:
        fixed_setup["actfunction"] = setup["actfunction"]
    except:
        fixed_setup["actfunction"] = "tanh"
        default_settings.append("actfunction")
    try:
        fixed_setup["architecture"] = setup["architecture"]
    except:
        fixed_setup["architecture"] = [5, 5]
        default_settings.append("architecture")
    try:
        fixed_setup["batch_size"] = setup["batch_size"]
    except:
        fixed_setup["batch_size"] = 16
        default_settings.append("batch_size")
    try:
        fixed_setup["epochs"] = setup["epochs"]
    except:
        fixed_setup["epochs"] = 5000
        default_settings.append("epochs")
    try:
        fixed_setup["initializer"] = setup["initializer"]
    except:
        fixed_setup["initializer"] = 'glorot_uniform'
        default_settings.append("initializer")
    try:
        fixed_setup["optimizer"] = setup["optimizer"]
    except:
        fixed_setup["optimizer"] = "adam"
        default_settings.append("optimizer")
    try:
        fixed_setup["optimizer_lr"] = setup["optimizer_lr"]
    except:
        fixed_setup["optimizer_lr"] = None
        default_settings.append("optimizer_lr")
    try:
        fixed_setup["data_augmentation"] = setup["data_augmentation"]
    except:
        fixed_setup["data_augmentation"] = False
        default_settings.append("data_augmentation")

    # Check if the setup dictionary has some unrecognised keys
    if len(setup) + len(default_settings) != len(fixed_setup):
        error('Error: unrecognised keys in the setup dictionary!')

    # Add the default values
    fixed_setup["default_settings"] = default_settings

    return fixed_setup

def get_optimizer(setup):
    """
    Return the optimizer specified in the "setup" dictionary.
    Optional keys:
        - setup["optimizer"]: optimizer in string format (default "adam")
        - setup["optimizer_lr"]: the learning rate (float >= 0)
    """

    # Return the right optimizer
    if setup['optimizer_lr'] is None:
        if setup["optimizer"] == "sgd":
            optimizer = SGD()
        elif setup["optimizer"] == "rmsprop":
            optimizer = RMSprop()
        elif setup["optimizer"] == "adagrad":
            optimizer = Adagrad()
        elif setup["optimizer"] == "adadelta":
            optimizer = Adadelta()
        elif setup["optimizer"] == "adam":
            optimizer = Adam()
        elif setup["optimizer"] == "adamax":
            optimizer = Adamax()
        elif setup["optimizer"] == "nadam":
            optimizer = Nadam()
    else:
        if setup["optimizer"] == "sgd":
            optimizer = SGD(lr=setup['optimizer_lr'])
        elif setup["optimizer"] == "rmsprop":
            optimizer = RMSprop(lr=setup['optimizer_lr'])
        elif setup["optimizer"] == "adagrad":
            optimizer = Adagrad(lr=setup['optimizer_lr'])
        elif setup["optimizer"] == "adadelta":
            optimizer = Adadelta(lr=setup['optimizer_lr'])
        elif setup["optimizer"] == "adam":
            optimizer = Adam(lr=setup['optimizer_lr'])
        elif setup["optimizer"] == "adamax":
            optimizer = Adamax(lr=setup['optimizer_lr'])
        elif setup["optimizer"] == "nadam":
            optimizer = Nadam(lr=setup['optimizer_lr'])
    
    return optimizer