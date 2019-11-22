# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import numpy as np

def chi2(data_A, data_B, errors2, weights=None, dof=None):
    
    # Set default value
    if weights is None:
        weights = np.ones(data_A.shape)
    if dof is None:
        dof = np.sum(weights != 0) # num of non-zero weights

    # Calculate the chi2
    return np.sum(np.square(weights * (data_A - data_B)) / errors2) / dof