# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import os
import matplotlib.pyplot as plt
from tools import make_dir, show
from mcnntunelib.templates.index import index
from mcnntunelib.templates.raw import raw
from mcnntunelib.templates.minimization import minimization
from mcnntunelib.templates.model import model
from mcnntunelib.templates.config import config
from jinja2 import Template


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

    def plot_minimize(self, minimizer):
        """"""
        minimizer.plot()
        plt.savefig('%s/plots/minimizer.svg' % self.path)

    def plot_model(self, model):
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