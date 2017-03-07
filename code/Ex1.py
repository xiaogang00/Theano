#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 11:27:09 2017

@author: root
"""

import theano
a = theano.tensor.vector()
b = theano.tensor.vector()

out = a**2 + b**2 + 2*a*b
f = theano.function([a,b], out)
print(f([0,1,2],[0,1,3]))