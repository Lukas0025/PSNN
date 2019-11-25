#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  reshape.py
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
reshape layer

@input ND numpy arrays
@output ND numpy array with shape (num of elements) -> with specific shape

SAMPLE:
input-shape: (10, 2)
reshape((-1,10))
output-shape(2, 10)
'''
class reshape:
  '''
  @param object self
  @param outshape - shape what you want on output
  '''
  def __init__(self, outshape):
      self.outshape = outshape

  '''
  @param object self
  @param 1D array of int inputshape
  '''
  def __create__(self, inputshape):
      #create test data with shape
      testdata = np.zeros(inputshape).reshape(self.outshape)

      return list(testdata.shape)

  '''
  @param object self
  @param ND array of float inputs
  '''
  def __forward__(self, inputs):
      return inputs.reshape(self.outshape)