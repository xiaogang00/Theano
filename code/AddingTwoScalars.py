#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:12:00 2017

@author: root
"""

import numpy
import theano.tensor as T
from theano import function

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)
print numpy.allclose(z.eval({x : 16.3, y : 12.1}), 28.4)
print numpy.allclose(f(16.3,12.1),28.4)