# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import os, copy
import matplotlib.pyplot as plt
from tools import make_dir, show
from mcnntunelib.templates.index import index
from mcnntunelib.templates.raw import raw
from mcnntunelib.templates.minimization import minimization
from mcnntunelib.templates.model import model
from mcnntunelib.templates.config import config
from jinja2 import Template
import numpy as np


class Report(object):

    def __init__(self, path):
        self.path = path
        make_dir('%s/plots' % self.path)

    def save(self, dictionary):
        tmplindex = [('index', Template(index)),
                     ('model', Template(model)),
                     ('minimization', Template(minimization)),
                     ('raw', Template(raw)),
                     ('config', Template(config))]

        for name, tmpl in tmplindex:
            output = tmpl.render(dictionary)
            with open('%s/%s.html' % (self.path, name), 'w') as f:
                f.write(output)

        show('\n- Generated report @ file://%s/%s/index.html' % (os.getcwd(), self.path))

    def plot_minimize(self, minimizer, best_x_unscaled, best_x_scaled, runs):
        """"""
        minimizer.es.plot()
        plt.savefig('%s/plots/minimizer.svg' % self.path)

        # plot 1d profiles
        for dim in range(runs.x_scaled.shape[1]):
            d = np.linspace(np.min(runs.x_scaled[:,dim]), np.max(runs.x_scaled[:,dim]), 30)
            res = []
            xx = []
            for p in d:
                a = np.array(best_x_scaled)
                a[dim] = p
                chi2 = minimizer.chi2(a)
                res.append(chi2)
                xx.append(runs.unscale_x(a)[dim])
            plt.figure()
            plt.plot(xx, res, label='parameter variation', linewidth=2)
            plt.axvline(best_x_unscaled[dim], color='r', linewidth=2, label='best value')
            plt.legend(loc='best')
            plt.title('1D profiles for parameter %d - %s' % (dim, runs.params[dim]))
            plt.ylabel('$\chi^2$/dof')
            plt.xlabel('parameter')
            plt.grid()
            plt.savefig('%s/plots/chi2_%d.svg' % (self.path, dim))

    def plot_model(self, model, runs):
        """"""
        plt.figure()
        plt.title('Training loss function vs iteration')
        plt.plot(model.loss, label='Loss')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.ylabel('ERF')
        plt.xlabel('iteration')
        plt.grid()
        plt.savefig('%s/plots/loss.svg' % self.path)


