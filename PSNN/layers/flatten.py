#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## @package flatten.py
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# flatten layer

import numpy as np

## flatten layer
# 
# @input ND numpy arrays
# @output ND numpy array with shape (num of elements) -> 1D
# 
# SAMPLE:
# input-shape: (10, 2)
# flatten()
# output-shape(20)
class flatten:

  ## 
  # @param object self
  # @param int size
  def __init__(self):
      pass

  ## 
  # @param object self
  # @param 1D array of int inputshape
  def __create__(self, inputshape):
      outshape = [1]

      for i in range(len(inputshape)):
        outshape[0] *= inputshape[i]

      return outshape

  ##
  # @param object self
  # @param ND array of float inputs
  def __forward__(self, inputs):
      return inputs.flatten()