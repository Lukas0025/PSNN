#!/usr/bin/env python
# -*- coding: utf-8 -*-
## @package PSNN.activation.relu
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# relu activation function

import numpy as np

class relu:
  def __init__(self, a = 0.001):
      self.a = a

  def __calc__(self, x):
      return np.where(x < 0, x, x * self.a)

  def __derivative__(self, x):
      out = np.where(x > 0, x, 1)
      return np.where(out <= 0, out, 0)