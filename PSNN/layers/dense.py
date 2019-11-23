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
@output ND numpy array with shape (NUM OF NEURONS, ...all input shape numbers without the first)

SAMPLE:
input-shape: (10, 10, 2)
dense(3)
output-shape(3, 10, 2) 
'''
class dense:
  '''
  This function be called when code create layer it define 
  how many neurons is in this layer is, what act function neurons
  using and what bias number be used
  
  @param object self
  @param int neurons - number of neurons (required)
  @param object activate - Instance of activation function class (default is linear) 
  @param float bias - bias input number (default is 1)
  @return None
  '''
  def __init__(self, neurons, activate = None, bias=1):
      self.neurons = neurons
      self.bias = bias
      # when activate is none use defaultactivation
      if activate == None:
        self.activate = defaultactivation()
      else:
        self.activate = activate

  '''
  This function has be called when network model
  is being formed it create array for weights and randomize it.
  it also defines what the output shape will be

  @param object self
  @param array inputshape - shape of input array
  @return array - shape of output array
  '''
  def __create__(self, inputshape):
      # clac weight shape for this input shape [down] [self.shape == shape of weights array]
      self.shape = inputshape[:]
      # because BIAS :D [bias is input too in every sum]
      self.shape[0] += 1
      # insert num of neurons as first shape [every neuron need weight for every input]
      self.shape.insert(0, self.neurons)
      # randomize Weights
      self.randomizeWeights()

      # create output shape
      inputshape[0] = self.neurons
      self.outputshape = inputshape[:]

      return inputshape

  '''
  This function randomize weights
  all of memory will be lost
  @param object self
  '''
  def randomizeWeights(self):
      self.weights = np.random.random_sample(self.shape)

  '''
  This function has be called when network model do prediction.
  It do prediction with inputs and return predicted values

  @param object self
  @param numpy array inputs - array with input data with correct shape
  @return numpy array - predicted values
  '''
  def __forward__(self, inputs):
      #if input.shape != self.shape[1:]:
      #  raise ValueError('Shape for dense layer is incorrent')

      # create array with bias we need one bias for one sum
      bias = np.full(self.shape[2:], self.bias)
      # add bias to input array
      inputs = np.append(inputs, [bias], axis = 0) 

      # init output array
      out = np.zeros(self.outputshape)

      # multiply inputs with theier weights for every neuron 
      for i in range(self.neurons):
        out[i] = np.sum(inputs * self.weights[i], axis = 0)

      # use activition function and return prediction
      return self.activate.__calc__(out)  
  
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

    # create array with bias we need one bias for one sum
    bias = np.full(self.shape[2:], self.bias)
    # add bias to input array
    inputs = np.append(inputs, [bias], axis = 0) 

    # calc previous layer error
    for i in range(len(inputs) - 1): # do not pass bias here (it is not from previous layer)
      nextfail.append(
        np.sum(fail * self.weights.T[i]) * self.activate.__derivative__(inputs[i])
      )

    # correct weights
    # inputs * fail * activate.derivative(output)
    for i in range(self.neurons):
      self.weights[i] = self.weights[i] + inputs * fail[i] * self.activate.__derivative__(output[i]) * rate

    return np.array(nextfail)

  '''
  This function has be called when you what randomly mutate layer.
  It add random number to every weight in range(-rate, rate).
  for random number use Normal distribution

  @param object self
  @param rate float - range of random number
  @return none
  '''
  def __mutate__(self, rate = 0.5):
      self.weights = self.weights + np.random.randn(*self.shape) * 0.20 * rate
