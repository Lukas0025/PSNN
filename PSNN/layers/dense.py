#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## @package PSNN.layers.dense
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# dense layer

import numpy as np
from ..activation import linear as defaultactivation

## dense layer (fully connect)
#
# input ND numpy arrays
# output ND numpy array with shape (NUM OF NEURONS, ...all input shape numbers without the first)
# 
# SAMPLE:
# input-shape: (10, 10, 2)
# dense(3)
# output-shape(3, 10, 2) 
class dense:
  ##
  # This function be called when code create layer it define 
  # how many neurons is in this layer is, what act function neurons
  # using and what bias number be used
  # 
  # @param self object
  # @param neurons number of neurons (required)
  # @param activate Instance of activation function class (default is linear) 
  # @param bias bias input number (default is 1)
  # @return None
  def __init__(self, neurons, activate = None, bias=1):
    self.neurons = neurons
    self.bias = bias
    # when activate is none use defaultactivation
    if activate == None:
      self.activate = defaultactivation()
    else:
      self.activate = activate

  ## This function has be called when network model
  # is being formed it create array for weights and randomize it.
  # it also defines what the output shape will be
  # 
  # @param self object
  # @param inputshape shape of input array
  # @return shape of output array
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

  ##
  # This function randomize weights
  # all of memory will be lost
  # @param self object
  def randomizeWeights(self):
    self.weights = np.random.random_sample(self.shape)

  ## This function has be called when network model do prediction.
  # It do prediction with inputs and return predicted values
  # 
  # @param self object
  # @param inputs array with input data with correct shape
  # @return redicted values
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
  
  ## This function has be called when network model do backpropagation learning.
  # It correct weights by error. It calc error of previous layer too.
  # 
  # @param self object
  # @param inputs array with input data with correct shape
  # @param output array with output data (what layer predicted) with correct shape
  # @param fail array with fail (base: target - output) with correct shape
  # @param rate rate value (size of correction jump)
  # @return error of previous layer
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

  ##
  # This function has be called when you what randomly mutate layer.
  # It add random number to every weight in range(-rate, rate).
  # for random number use Normal distribution
  # 
  # @param self object
  # @param rate range of random number
  # @return none
  def __mutate__(self, rate = 0.5):
    self.weights = self.weights + np.random.randn(*self.shape) * 0.20 * rate
