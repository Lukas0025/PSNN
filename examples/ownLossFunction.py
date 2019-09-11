#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PYSNN import model, layers, activation

def myloss(outputs, targets):
	return abs(np.sum(targets - output))


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
		lossfunc=myloss,
		epochs=50
)

print("predict after learn: " + str(netmodel.predict(testin[0])))
print("real traget:" + str(testout[0]))