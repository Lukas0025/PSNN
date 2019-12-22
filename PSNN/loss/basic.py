#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## @package basic.py
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# Basic loss functions L1 (mae) and L2 (mse)

import numpy as np

## Mean absolute error (L1)
# error calulator target - output
class mae():
  ## calc error for every output
  # @param object self
  # @param numpy array outputs - outputs from neuron network
  # @param numpy array targets - target what you want 
  # @return numpy array with errors
  def __calc__(self, outputs, targets):
    return targets - outputs

  ## calc sum of errors for every output (one value error)
  # @param object self
  # @param numpy array outputs - outputs from neuron network
  # @param numpy array targets - target what you want 
  # @return float error
  def __clac1V__(self, outputs, targets):
    return np.sum(np.abs(self.__calc__(outputs, targets)))

## Mean squared error (L2)
# error calulator (target - output) ** 2
class mse():
  ## calc error for every output
  # @param object self
  # @param numpy array outputs - outputs from neuron network
  # @param numpy array targets - target what you want
  # @return numpy array with errors
  def __calc__(self, outputs, targets):
    oriantion = np.where(targets - outputs < 0, targets - outputs, 1)
    oriantion = np.where(oriantion >= 0, oriantion, -1)

    return (targets - outputs) ** 2 * oriantion

  ## calc sum of errors for every output (one value error)
  # @param object self
  # @param numpy array outputs - outputs from neuron network
  # @param numpy array targets - target what you want 
  # @return float error
  def __clac1V__(self, outputs, targets):
    return np.sum(np.abs(self.__calc__(outputs, targets)))