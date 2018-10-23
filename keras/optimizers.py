# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 10/16/2018 3:03 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
from utils.theano_utils import shared_zeros, shared_scalar

class Optimizer(object):
    def get_updates(self):
        raise  NotImplementedError

    def get_gradients(self, cost, params):
        grads = T.grad(cost, params)

        return grads

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, *args, **kwargs):
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)

    def get_updates(self, params, cost):
        grads = self.get_gradients(cost, params)
        lr = self.lr - self.decay * self.iterations
        updates = [(self.iterations, self.iterations + 1.)]

        for p, g in zip(params, grads):
            m = shared_zeros(p.get_value().shape)  # momentum
            v = self.momentum * m - lr * g  # velocity
            updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            updates.append((p, new_p))
        return updates


sgd = SGD

from utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'optimizer', instantiate=True)