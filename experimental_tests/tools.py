#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some shared tools for nnfit
"""

__author__  = "Stefano Carrazza et al."

import numpy as np
import pandas as pd
import pickle


def standardize(x, store=False, file=None):
    """apply standardization and store to file mean and std dev."""
    xm = np.mean(x)
    st = np.std(x)
    if store:
        with open(file, 'wb') as f:
            f.write('%.5e\t%.5e' % (xm,st))
    return (x-xm)/st


def chi2(y_pred, y_true, invcov):
    """The chi-squared definition"""
    X = y_pred-y_true
    return X.dot(invcov.dot(X))


def make_dir(dir_name):
    """Creates directory"""
    try:
        import os, sys
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        elif not os.path.isdir(dir_name):
            print('output is not a directory')
            sys.exit(1)
    except:
        pass

def load_data(inputfile):
    data = pd.read_csv(inputfile, sep='\t', skiprows=1)
    tg = np.array(data['target'])
    tg_error = np.array(data['error'])
    x = []
    for i in range(100):
        try:
            x.append( np.array(data['x%d' % i]) )
        except KeyError:
            break
    return x, tg, tg_error


def load(input_folder, max_reps):
    """Load model and weights from file"""
    from keras.models import load_model
    hists = []
    models = []
    for i in range(1,max_reps):
        try:
            hists.append(pickle.load( open('%s/hist-%d.p' % (input_folder, i), "rb")))
            models.append(load_model('%s/weights-%d.hdf5' % (input_folder,i)))
            print('[load] loaded replica %d' % i)
        except:
            print('[load] problem with replica %d' % i)
    return hists, models


def validate(chi2todata, loss, val, stops, models, plt):
    """produce validation plots"""
    plt.figure()
    plt.hist(chi2todata, bins=50, alpha=0.5);
    plt.title(r'$\chi^2$ per replica')
    plt.xlabel(r'$\chi^2$')

    plt.figure()
    plt.hist(loss, bins=50, alpha=0.5, label='training');
    plt.hist(val, bins=50, alpha=0.5, label='validation');
    plt.title('Cross-validation quality')
    plt.xlabel(r'$\chi2$')
    plt.legend()

    plt.figure()
    plt.scatter(loss, val, alpha=0.5);
    plt.title('Cross-validation scatter')
    plt.xlabel('loss')
    plt.ylabel('validation')

    plt.figure()
    plt.hist(stops, weights=[1./len(models)]*len(stops), alpha=0.5);
    plt.title('Stopping points')
    plt.xlabel('Epochs')
    plt.ylabel('Fraction')


def postfit(hist, models, nsigma=4):
    """filter replicas by nsigma"""
    nsigma = np.abs(nsigma)
    chi2s = [v['chi2'] for v in hist]
    mean = np.mean(chi2s)
    std = np.std(chi2s)
    bad = False
    for c in chi2s:
        if c > mean+nsigma*std: bad = True
    while bad:
        drops = []
        chi2s = [v['chi2'] for v in hist]
        mean = np.mean(chi2s)
        std = np.std(chi2s)
        for rep in range(len(models)):
            if hist[rep]['chi2'] > mean+nsigma*std or hist[rep]['chi2'] < mean-nsigma*std:
                print('[postfit] dropping replica %d with chi2=%f' % (rep, hist[rep]['chi2']))
                drops.append(rep)
        drops.sort()
        drops = drops[::-1]
        for d in drops:
            hist.pop(d)
            models.pop(d)
        if len(drops) == 0: bad = False
    return hist, models


def export_plain(model, fileout):
    """export neural net to plain text"""
    import json
    arch = json.loads(model.to_json())
    with open(fileout, 'w') as fout:
        fout.write('layers ' + str(len(model.layers)) + '\n')
        for ind, l in enumerate(arch["config"]):
            fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')
            fout.write('activation ' + l['config']['activation'] + '\n')
            if l['class_name'] == 'Dense':
                W = model.layers[ind].get_weights()[0]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                for w in W:
                    fout.write(str(w) + '\n')
                fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
            else:
                print('[export_plain] model not implemented!')
