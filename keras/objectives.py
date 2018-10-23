# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 10/23/2018 3:14 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np

def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

# aliases
mse = MSE = mean_squared_error

from utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')

# def to_categorical(y):
#     '''Convert class vector (integers from 0 to nb_classes)
#     to binary class matrix, for use with categorical_crossentropy
#     '''
#     nb_classes = np.max(y)+1
#     Y = np.zeros((len(y), nb_classes))
#     for i in range(len(y)):
#         Y[i, y[i]] = 1.
#     return Y
