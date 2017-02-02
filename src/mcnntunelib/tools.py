# -*- coding: utf-8 -*-
"""
Generates custom runcards for tunes variations
"""
__author__ = "Stefano Carrazza & Simone Alioli"
__version__= "1.0.0"


def make_dir(dir_name):
    """Creates directory"""
    try:
        import os, sys
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        elif not os.path.isdir(dir_name):
            raise Exception('Output is not a directory')
    except:
        raise Exception('Error creating output folder.')
