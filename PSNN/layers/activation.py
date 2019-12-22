#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## @package PSNN.layers.activation
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# Activation layer

import numpy as np
from ..activation import linear as defaultactivation

## activation layer
#
# @input ND numpy arrays
# @output ND numpy array with same shape
# 
# SAMPLE:
# input-shape: (10, 10, 2)
# activation()
# output-shape(10, 10, 2) 
class activation:
  ## init layer
  # @param self object
  # @param activate activate class
  # @return None
  def __init__(self, activate = None):
    if activate == None:
      self.activate = defaultactivation()
    else:
      self.activate = activate

  ## create layer
  # @param object self
  # @param inputshape 1D array of int
  # @return spahe of output from this layer
  def __create__(self, inputshape):
    return inputshape

  ## clac forward on layer
  # @param self object
  # @param inputs ND array of float inputs
  # @return forwarded input (ND numpy array)
  def __forward__(self, inputs):
    return self.activate.__calc__(inputs)

  ## do backpropagation
  # @param object self
  # @param inputs ND array of float inputs 
  # @param output ND array of float outputs
  # @param fail ND array of float fail (errors) of layers
  # @param rate float of size of correction
  # @return fail of previous layer (numpy array)
  def __backprop__(self, inputs, output, fail, rate):
    nextfail = []

    for i in range(len(fail)):
      nextfail.append(self.activate.__derivative__(output[i]) * fail[i])

    return np.array(nextfail)

