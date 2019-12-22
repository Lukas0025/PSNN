#!/usr/bin/env python
# -*- coding: utf-8 -*-
## @package PSNN.activation.linear
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# linear activation function

import numpy as np

class linear:
  def __calc__(self, x):
      return x
  
  def __derivative__(self, x):
      return np.ones(x.shape)