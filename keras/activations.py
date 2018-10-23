# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 10/23/2018 2:49 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theano
import theano.tensor as T
import types

# def softmax(x):
#     return K.softmax(x)

def softmax(x):
    return T.nnet.softmax(x)

def relu(x):
    return (x + abs(x)) / 2.0

def linear(x):
    return x

from utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'activation function')