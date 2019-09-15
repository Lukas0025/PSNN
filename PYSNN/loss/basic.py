#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  
#  basic loss functions (L1 and L2)
#  basic.py
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

import numpy as np

class mae():
  def __calc__(self, outputs, targets):
    return targets - outputs

  def __clac1V__(self, outputs, targets):
    return np.sum(np.abs(self.__calc__(outputs, targets)))

class mse():
  def __calc__(self, outputs, targets):
    oriantion = np.where(targets - outputs < 0, targets - outputs, 1)
    oriantion = np.where(oriantion >= 0, oriantion, -1)

    return (targets - outputs) ** 2 * oriantion

  def __clac1V__(self, outputs, targets):
    return np.sum(np.abs(self.__calc__(outputs, targets)))