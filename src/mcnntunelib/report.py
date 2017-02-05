# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import os
from tools import make_dir, show
from mcnntunelib.templates.index import index
from mcnntunelib.templates.raw import raw
from mcnntunelib.templates.config import config
from jinja2 import Template


class Report(object):

    def __init__(self, path):
        self.path = path

    def save(self, dictionary):
        make_dir('%s/plots' % self.path)
        tmplindex = [('index', Template(index)),
                     ('raw', Template(raw)),
                     ('config', Template(config))]

        for name, tmpl in tmplindex:
            output = tmpl.render(dictionary)
            with open('%s/%s.html' % (self.path, name), 'w') as f:
                f.write(output)

        show('\n- Generated report @ file://%s/%s/index.html' % (os.getcwd(), self.path))