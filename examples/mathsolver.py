#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#use evolutionFit to solve math problem

import numpy as np
from PSNN import model, layers

netmodel = model([
  layers.dense(1)
])

inaccuracy = 1e-14

def mycubic(model):
    # 2*x*x*x + 5*x*x = 10

    x = model.predict([])[0]
    loss = abs(10 - (2*x*x*x + 5*x*x))

    return loss

netmodel.create(
  inputs=(0,)
)

drate = 1
lastlost = 0

while True:
    loss = netmodel.evolutionFit(
        rate=drate,
        replication=200,
        lossfunc=mycubic,
        epochs=10
    )

    if loss == lastlost:
        drate = drate / 10
    else:
        lastlost = loss

    if (loss <= 0 + inaccuracy):
        break;
    
print("I found one of solution:")
print(netmodel.predict([]))
