#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  relu.py
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

class relu:
  def __init__(self, a = 0.001):
      self.a = a

  def __calc__(self, x):
      return np.where(x < 0, x, x * self.a)

  def __derivative__(self, x):
      out = np.where(x > 0, x, 1)
      return np.where(out <= 0, out, 0)