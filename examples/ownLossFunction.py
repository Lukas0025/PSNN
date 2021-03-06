#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PSNN import model, layers, activation

class myloss:
  # this is calc loss normaly
  # @param outputs np array
  # @param targets np array
  def __calc__(self, outputs, targets):
    return targets - outputs

  # this return one value loss must by abs
  # @param outputs np array
  # @param targets np array
  def __clac1V__(self, outputs, targets):
    return np.absolute(np.sum(self.__calc__(outputs, targets)))


netmodel = model([
  layers.dense(2, activation.sigmoid()),
  layers.dense(1, activation.sigmoid())
])

netmodel.debug = True

netmodel.create(
  inputs=(4,)
)

testin = np.random.random((1000, 4))
testout = np.random.random((1000, 1))

print("predict output before learn: " + str(netmodel.predict(testin[0])))

# learn by test in and out
netmodel.fit(
  inputs=testin,
  targets=testout,
  rate=1,
  replication=20,
  lossfunc=myloss(),
  epochs=50
)

print("predict after learn: " + str(netmodel.predict(testin[0])))
print("real traget:" + str(testout[0]))
