# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
"""
import numpy as np
import tensorflow as tf


def chi2(data_A, data_B, errors2, weights=None, dof=None):

    # Set default value
    if weights is None:
        weights = np.ones(data_A.shape)
    if dof is None:
        dof = np.sum(weights)

    # Calculate the chi2
    return np.sum(np.square(weights * (data_A - data_B)) / errors2) / dof


def chi2_tf(data_A, data_B, errors2, weights=None, dof=None):

    # Set default value
    if weights is None:
        weights = np.ones(data_A.shape)
    if dof is None:
        dof = np.sum(weights)

    # Calculate the chi2
    return tf.math.reduce_sum(tf.math.square(weights * (data_A - data_B)) / errors2) / dof
