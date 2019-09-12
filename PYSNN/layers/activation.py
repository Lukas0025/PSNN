#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  activation.py
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
activation layer

@input ND numpy arrays
@output ND numpy array with same shape

SAMPLE:
input-shape: (10, 10, 2)
activation()
output-shape(10, 10, 2) 
'''
class activation:
  '''
  @param object self
  @param class activate
  '''
  def __init__(self, activate = None):
      if activate == None:
        self.activate = defaultactivation()
      else:
        self.activate = activate

  '''
  @param object self
  @param 1D array of int inputshape
  '''
  def __create__(self, inputshape):
      return inputshape

  '''
  @param object self
  @param ND array of float inputs
  '''
  def __forward__(self, inputs):
      return self.activate.__calc__(inputs)

  def __backprop__(self, inputs, output, fail, rate):
    nextfail = []

    for i in range(len(fail)):
      nextfail.append(self.activate.__derivative__(output[i]) * fail[i])

    return np.array(nextfail)

