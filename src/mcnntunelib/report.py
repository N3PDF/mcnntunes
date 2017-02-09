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
from mcnntunelib.templates.data import data
from jinja2 import Template
import numpy as np


class Report(object):

    def __init__(self, path):
        self.path = path
        make_dir('%s/plots' % self.path)

    def save(self, dictionary):
        tmplindex = [('index', Template(index)),
                     ('data', Template(data)),
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

    def plot_model(self, model, runs, data):
        """"""
        if not model.use_scan:
            plt.figure()
            plt.title('Training loss function vs iteration')
            plt.plot(model.loss, label='Loss')
            plt.legend(loc='best')
            plt.yscale('log')
            plt.ylabel('ERF')
            plt.xlabel('iteration')
            plt.grid()
            plt.savefig('%s/plots/loss.svg' % self.path)

        # scatter plot
        loss_runs = []
        for r in range(runs.x_scaled.shape[0]):
            loss_runs.append(model.model.evaluate(runs.x_scaled[r].reshape(1,runs.x_scaled.shape[1]),
                                        runs.y_scaled[r].reshape(1,runs.y_scaled.shape[1]),
                                        verbose=0))
        plt.figure()
        plt.plot(loss_runs)
        plt.axhline(model.loss[-1], color='r', linewidth=2, label='average')
        plt.legend(loc='best')
        plt.xlabel('variations')
        plt.ylabel('MSE')
        plt.title('Loss function for final model')
        plt.grid()
        plt.savefig('%s/plots/model_loss.svg' % self.path)

        plt.figure()
        plt.hist(loss_runs, color='g')
        plt.axvline(model.loss[-1], color='r', linewidth=2, label='average')
        plt.xlabel('MSE')
        plt.ylabel('#')
        plt.title('Loss function distribution')
        plt.grid()
        plt.savefig('%s/plots/model_loss_hist.svg' % self.path)

    def plot_data(self, data, predictions, runs):

        ifirst = 0
        for i, hist in enumerate(data.plotinfo):
            size = len(hist['y'])
            plt.figure()
            plt.subplot(211)

            rs = [ item for item in runs.plotinfo if item['title'] == hist['title']]
            up = np.vstack([r['y'] for r in rs]).max(axis=0)
            dn = np.vstack([r['y'] for r in rs]).min(axis=0)

            plt.fill_between(hist['x'], dn, up, color='#ffff00')

            plt.errorbar(hist['x'], hist['y'], yerr=hist['yerr'],
                         marker='o', linestyle='none', label='data')

            plt.plot(hist['x'], predictions[ifirst:ifirst+size], '-', label='best model')

            plt.yscale('log')
            plt.legend(loc='best')
            plt.title(hist['title'])

            plt.subplot(212)

            plt.errorbar(hist['x'], hist['y']/hist['y'], yerr=hist['yerr']/hist['y'],
                         marker='o', linestyle='none', label='data')
            plt.plot(hist['x'], predictions[ifirst:ifirst+size]/hist['y'], '-', label='best model')

            plt.savefig('%s/plots/%d_data.svg' % (self.path, i))

            ifirst += size
