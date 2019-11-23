#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PSNN import model, layers, activation

netmodel = model([
  layers.dense(2, activation.sigmoid()),
  layers.dense(1, activation.sigmoid())
])

netmodel.debug = True

netmodel.create(
  inputs=(3,)
)

inputs = np.array([
  [0, 0, 1], #0
  [0, 0, 0], #0
  [1, 0, 0], #1
  [1, 1, 1], #1
  [0, 1, 0], #1
  [0, 1, 1] #1
])

targets = np.array([
  [0],
  [0],
  [1],
  [1],
  [1],
  [1],
])

netmodel.fit(
  rate=1,
  inputs=inputs,
  targets=targets,
  replication=200,
  epochs=50
)

# try predict 1 1 0 -> 1
print(netmodel.predict([1,1,0]))
