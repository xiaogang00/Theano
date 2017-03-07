#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:00:16 2017

@author: root
"""

import numpy
import theano
import theano.tensor as T
from theano import pp
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
pp(gy)  # print out the gradient prior to optimization
#'((fill((x ** TensorConstant{2}), TensorConstant{1.0}) * TensorConstant{2}) * (x ** (TensorConstant{2} - TensorConstant{1})))'
f = theano.function([x], gy)
f(4)
#array(8.0)
numpy.allclose(f(94.2), 188.4)
#True

##-------------------------
x = T.dmatrix('x')
s = T.sum(1/(1+T.exp(-x)))
gs = T.grad(s, x)
dlogistic = theano.function([x], gs)
dlogistic([[0,1], [-1,-2]])
 