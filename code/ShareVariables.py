#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:07:23 2017

@author: root
"""

import theano.tensor as T
from theano import function
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
decrementor = function([inc], state, updates=[(state, state-inc)])

fn_of_state = state * 2 + inc
# The type of foo must match the shared variable we are replacing with the ``givens`
foo = T.scalar(dtype=state.dtype)
# The givens parameter can be used to replace any symbolic variable, not just a shared 
# variable. You can replace constants, and expressions, in general. 
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print skip_shared(1,3)
print state.get_value()