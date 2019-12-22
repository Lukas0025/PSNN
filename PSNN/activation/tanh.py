#!/usr/bin/env python
# -*- coding: utf-8 -*-
## @package PSNN.activation.tanh
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# tanh activation function

import numpy as np

class tanh:
  def __calc__(self, x):
      return np.tanh(x)

  def __derivative__(self, x):
      return 1 - self.__calc__(x) ** 2