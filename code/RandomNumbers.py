#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:00:34 2017

@author: root
"""

## Brief Example

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True) # Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2*rv_u)

f_val0 = f()
f_val1 = f()  #different numbers from f_val0

g_val0 = g()  # different numbers from f_val0 and f_val1
g_val1 = g()  # same numbers as g_val0!

nearly_zeros() # exactly 0

## Seeding Streams
#-----------------------------------------#
## TODO
#-----------------------------------------#