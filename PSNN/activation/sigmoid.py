#!/usr/bin/env python
# -*- coding: utf-8 -*-
## @package PSNN.activation.sigmoid
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# sigmoid activation function 

import numpy as np

class sigmoid:
  def __calc__(self, x):
      return 1 / (1 + np.exp(-x))

  def __derivative__(self, x):
      return self.__calc__(x) * (1 - self.__calc__(x))
