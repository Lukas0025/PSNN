#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  dense.py
#  
#  Copyright 2019 Lukáš Plevač <lukasplevac@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#

import numpy as np
from ..activation import linear as defaultactivation

'''
dense layer (fully connect)

@input ND numpy arrays
@output ND numpy array with shape concat of all branches

SAMPLE:
input-shape: (2,)
concat([
    [
        dense(3)
    ],

    [
        dense(3)
    ]
])
output-shape(6,) 
'''
class concat:
    '''
    This function be called when code create layer it define branches
  
    @param object self
    @param list of branches - list what have lists of layers
    @return None
    '''
    def __init__(self, branches):
        self.branches = branches

    '''
    This function has be called when network model
    is being formed and it randomize kernel.
    it also defines what the output shape will be

    @param object self
    @param array inputshape - shape of input array
    @return array - shape of output array
    '''
    def __create__(self, inputshape):
        self.shape = inputshape[:]
        self.outshape = []
        realOutSahpe = []

        for branch in self.branches:
            out = inputshape[:]
            for layer in branch:
                out = layer.__create__(out)
            realOutSahpe = out
            self.outshape.append(out[0])
        
        realOutSahpe[0] = np.sum(self.outshape)
        return realOutSahpe

    '''
    This function has be called when network model do prediction.
    It do prediction with inputs and return predicted values

    @param object self
    @param numpy array inputs - array with input data with correct shape
    @return numpy array - predicted values
    '''
    def __forward__(self, inputs):
        outputs = []

        for branch in self.branches:
            out = inputs[:]
            for layer in branch:
                out = layer.__forward__(out)
            outputs.append(out)

        return np.concatenate(outputs)
  
    '''
    This function has be called when network model do backpropagation learning.
    It correct weights by error. It calc error of previous layer too.

    @param object self
    @param numpy array inputs - array with input data with correct shape
    @param numpy array output - array with output data (what layer predicted) with correct shape
    @param numpy array fail - array with fail (base: target - output) with correct shape
    @param float rate - rate value (size of correction jump)
    @return numpy array - error of previous layer
    '''
    def __backprop__(self, inputs, output, fail, rate):
        nextfail = []

        # first do forward
        outputs = []

        for branch in self.branches:
            outputs.append([inputs[:]])
            for layer in branch:
                outputs[-1].append(
                    layer.__forward__(outputs[-1][-1])
                )

        start = 0
        for i in range(len(self.branches)): 
            # split fail to branches
            localfail = fail[ start : start + self.outshape[i] ]
            start += self.outshape[i]

            for j in range(len(self.branches[i]) - 1, -1, -1):
                if hasattr(self.branches[i][j], '__backprop__'):
                    localfail = self.branches[i][j].__backprop__(outputs[i][j], outputs[i][j + 1], localfail, rate)

            nextfail.append(localfail)

        return np.sum(nextfail, axis = 0) / len(self.branches)

    '''
    This function has be called when you what randomly evolute layer.
    It evolute every layer in every branch 

    @param object self
    @param rate float - range of random number
    @return none
    '''
    def __evolute__(self, rate = 0.5):
        for branch in self.branches:
            for layer in branch:
                if hasattr(layer, '__evolute__'):
                    layer.__evolute__(rate)

