#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 11:20:54 2017

@author: root
"""

import numpy
import theano.tensor as T
from theano import *

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
f(numpy.array([[1, 2], [3, 4]]), numpy.array([[10, 20], [30, 40]]))