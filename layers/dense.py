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

'''
dense layer (fully connect)

@input ND numpy arrays
@output ND numpy array with shape (NUM OF NEURONS, ...all input shape numbers without the first)

SAMPLE:
input-shape: (10, 10, 2)
dense(3)
output-shape(3, 10, 2) 
'''
class dense:
    '''
    @param object self
    @param int neurons
    @param class activate
    @param float bias
    '''
    def __init__(self, neurons, activate = None, bias=1):
        self.neurons = neurons
        self.bias = bias
        self.activate = activate

    '''
    @param object self
    @param 1D array of int inputshape
    '''
    def __create__(self, inputshape):
        #clac shape for this input schape
        self.shape = inputshape[:]
        self.shape.insert(0, self.neurons)
        #randomize Weights
        self.randomizeWeights()

        inputshape[0] = self.neurons
        self.outputshape = inputshape[:]

        return inputshape

    '''
    @param object self
    '''
    def randomizeWeights(self):
        self.weights = np.random.random_sample(self.shape)

    '''
    @param object self
    @param ND array of float inputs
    '''
    def __forward__(self, inputs):
        #if input.shape != self.shape:
        #    raise ValueError('Shape for dense layer is incorrent')

        return self.layerCalc(inputs)

    def layerCalc(self, inputs):
        out = np.zeros(self.outputshape)

        for i in range(self.neurons):
            out[i] = np.sum(inputs * self.weights[i], axis = 0)

        return self.activate.calc(self.activate, out)

    
    def __backprop__(self, error):
        pass

    '''
    @param object self
    @param rate float
    '''
    def __evolute__(self, rate = 0.5):
        self.weights = self.weights + np.random.uniform(-rate, rate, self.shape)

