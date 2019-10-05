#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PYSNN import model, layers, activation

netmodel = model([
  layers.convolve((2,2)),
  layers.flatten(),
  layers.dense(1, activation.sigmoid())
])

netmodel.debug = True

netmodel.create(
  inputs=(3,3)
)

inputs = np.array([
  [ # 1
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
  ],

  [ # 0
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
  ],

  [ # 0
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]
])

targets = np.array([
  [1],
  [0],
  [0]
])

# try predict
print(netmodel.predict(inputs[0]))

netmodel.fit(
  rate=1,
  inputs=inputs,
  targets=targets,
  replication=200,
  epochs=50
)

# try predict
print(netmodel.predict(inputs[0]))
