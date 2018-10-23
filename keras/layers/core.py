# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 10/16/2018 2:50 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theano
import theano.tensor as T

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, floatX

class Layer(object):
    def connect(self, previous_layer):
        self.previous_layer = previous_layer

    def output(self, train):
        raise NotImplementedError

    def get_input(self, train):
        if hasattr(self, 'previous_layer'):
            return self.previous_layer.output(train=train)
        else:
            return self.input

    def set_weights(self, weights):
        for p, w in zip(self.params, weights):
            p.set_value(floatX(w))

    def get_weights(self):
        weights = []
        for p in self.params:
            weights.append(p.get_value())
        return weights

class Activation(Layer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation):
        self.activation = activations.get(activation)
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        return self.activation(X)


class Flatten(Layer):
    '''
        Reshape input to flat shape.
        First dimension is assumed to be nb_samples.
    '''
    def __init__(self, size):
        self.size = size
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        nshape = (X.shape[0], self.size)
        return theano.tensor.reshape(X, nshape)

class Dense(Layer):
    '''
        Just your regular fully connecter NN layer.
    '''

    def __init__(self, input_dim, output_dim, init='uniform', activation='linear', weights=None):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output