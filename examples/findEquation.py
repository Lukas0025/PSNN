#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from PSNN import model, layers, activation

netmodel = model([
  layers.dense(2, activation.sigmoid()),
  layers.dense(2, activation.sigmoid()),
  layers.dense(1, activation.linear())
])

netmodel.debug = 1
netmodel.poolSize = 8 # i have 8 cores

netmodel.create(
  inputs=(2,)
)

inputs = []
targets = []

# genarate samples for nn
# we try find x1 ** 2 + x2 ** 2 = y ** 2
for i in range(10): # make 10 samples
  x = np.random.random(2,) * 7
  inputs.append(x)
  y = math.sqrt(np.sum(x ** 2))
  targets.append(y)


drate = 1 # dynamic rate
lastlost = 0
inaccuracy = 0.4

while True:
  loss = netmodel.evolutionFit(
    rate=drate,
    inputs=inputs,
    targets=targets,
    replication=28,
    epochs=1
  )

  if loss == lastlost:
    drate = drate / 10
  else:
    lastlost = loss

  if (loss <= inaccuracy):
     break

# try predict
x = np.random.random(2,) * 7
print(x)
print("I need {}".format(math.sqrt(np.sum(x ** 2))))
print(netmodel.predict(x))
