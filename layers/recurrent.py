#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  recurrent.py
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
recurrent layer

@input ND numpy arrays
@output ND numpy array with shape (MEM SIZE, ...all input shape numbers)

SAMPLE:
input-shape: (10, 10, 2)
recurrent(3)
output-shape(3, 10, 10, 2) 
'''
class recurrent:
  '''
  @param object self
  @param int size
  '''
  def __init__(self, size):
      self.size = size

  def __clrmem__(self):
      self.mem = []

      for i in range(self.size):
        self.mem.append(np.zeros(self.shape))

  def addToMem(self, data):
      del self.mem[-1]
      self.mem.insert(0, data)

  '''
  @param object self
  @param 1D array of int inputshape
  '''
  def __create__(self, inputshape):
      self.shape = inputshape[:]
      self.__clrmem__()

      inputshape.insert(0, self.size)
      return inputshape

  '''
  @param object self
  @param ND array of float inputs
  '''
  def __forward__(self, inputs):
      self.addToMem(inputs)
      return np.array(self.mem)