# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 10/23/2018 3:40 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros

class LeakyReLU(Layer):
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        return ((X + abs(X)) / 2.0) + self.alpha * ((X - abs(X)) / 2.0)