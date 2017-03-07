#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:26:19 2017

@author: root
"""

import theano
import theano.tensor as T
state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates=[(state, state+inc)])
print accumulator(10)
print state.get_value()

new_state = theano.shared(0)
new_accumulator = accumulator.copy(swap={state:new_state})
print new_accumulator(100)
print new_state.get_value()

# ERROR https://github.com/Theano/Theano/issues/4980

#null_accumulator = accumulator.copy(delete_updates=True)
#null_accumulator(9000)
#print state.get_value()