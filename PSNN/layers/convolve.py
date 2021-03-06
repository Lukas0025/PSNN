#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## @package PSNN.layers.convolve
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# convolve layer

import numpy as np


## convolve layer (convolution)
#
# input 2D numpy array
# output 2D numpy array -> shape is np.subtract(inputs.shape, kernel.shape) + 1
# 
# SAMPLE:
# input-shape: (10,10)
# convolve((2,2))
# output-shape: (9,9) 
class convolve:
  
  ## This function be called when code create layer it define 
  # size of kernel
  # @param self object
  # @param kSize size of kernel (filter) (required)
  # @return None
  def __init__(self, kSize):
    self.kSize = kSize


  ## do convolution on 2D array
  # @param object self
  # @param inputs inputs to convolve
  # @param kernel kernel for convolution 
  # @return convolved numpy array
  def convolve2d(self, inputs, kernel):
    # calc the size of the array of submatracies
    sub_shape = tuple(np.subtract(inputs.shape, kernel.shape) + 1)

    # alias for the function
    strd = np.lib.stride_tricks.as_strided

    # make an array of submatracies
    submatrices = strd(inputs,kernel.shape + sub_shape,inputs.strides * 2)

    # sum the submatraces and kernel
    convolved_matrix = np.einsum('ij,ijkl->kl', kernel, submatrices)

    return convolved_matrix

  
  ## This function has be called when network model
  # is being formed it randomize kernel too.
  # @param self object
  # @param inputshape shape of input array
  # @return shape of output array
  def __create__(self, inputshape):
    self.shape = inputshape[:]
    
    # calc output array size
    self.outShape = tuple(np.subtract(inputshape, self.kSize) + 1)
    
    # randomize kernel
    self.randomizeKernel()

    print(self.outShape)

    return list(self.outShape)

  ## This function randomize kernel
  # all of memory will be lost
  # @param self object
  # @return None
  def randomizeKernel(self):
    self.kernel = np.random.random_sample(self.kSize)

  ## This function has be called when network model do prediction.
  # It do prediction with inputs and return predicted values
  #
  # @param object self
  # @param inputs array with input data with correct shape
  # @return predicted values
  def __forward__(self, inputs):
    return self.convolve2d(inputs, self.kernel)

  
  ##
  # This function has be called when network model do backpropagation learning.
  # It correct kernel by error. It calc error of previous layer too.
  #
  # @param object self
  # @param inputs array with input data with correct shape
  # @param output array with output data (what layer predicted) with correct shape
  # @param fail array with fail (base: target - output) with correct shape
  # @param rate rate value (size of correction jump)
  # @return error of previous layer
  # @todo
  def __backprop__(self, inputs, output, fail, rate):
    raise Exception('convolve layer not support backprop learning now. sorry')

  
  ##This function has be called when you what randomly evolute layer.
  #It add random number to every weight in range(-rate, rate).
  #
  # @param self object
  # @param rate range of random number
  # @return none
  def __mutate__(self, rate = 0.5):
    self.kernel = self.kernel + np.random.uniform(-rate, rate, self.kSize)

