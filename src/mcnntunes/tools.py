# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
"""
import logging
log = logging.getLogger(__name__)


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

# message handlers
def show(message):
    print(message)
    log.info(message)


def info(message):
    print('\033[94m%s\033[0m' % message)
    log.info(message)


def success(message):
    print('\033[92m%s\033[0m' % message)
    log.info(message)


def error(message):
    log.error(message, exc_info=True)
    show('\n')
    raise Exception('\033[91m%s\033[0m' % message)

# check if basicConfig has been called
# if not, don't log anything
# this is a low-quality workaround
# for the hyperopt-mongo-workers
def log_check():
    if len(log.__dict__['parent'].__dict__['handlers']) == 2:
        log.disabled = True
