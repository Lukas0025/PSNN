#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import glob
from PIL import Image
from PSNN import model, layers, activation
  
# this get inputs arrays form images in directory
#
# @param directory str location of dir with images
# @param target np array of target (output) for images in this dir
# @return tuple of [0] -> images as inputs arrays [1] -> traget (output) array for same input
def getITArray(directory, target):
  imgs = glob.glob(directory + "/*.png")
  out = ([], [])
  for loc in imgs:
    img = Image.open(loc).convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size
    data = list(img.getdata()) # convert image data to a list of integers
    out[0].append([data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)])
    out[1].append(target)

  return out

netmodel = model([
  layers.flatten(),
  layers.activation(activation.tanh()),
  layers.dense(30, activation.tanh()),
  layers.dense(20, activation.tanh()),
  layers.dense(10, activation.sigmoid())
])

netmodel.debug = True

netmodel.create(
  inputs=(28,28)
)

# get learn data from images
learn = {
  "inputs": [],
  "targets": []
}

# fill imputs and targets
for i in range(10):
  # one hot encoding
  tar = np.zeros((10))
  tar[i] = 1
  # get inputs arrays and targets arrays from images
  inputs, targets = getITArray("mnist/training/" + str(i), tar)
  learn['inputs'] += inputs
  learn['targets'] += targets

# convert lists to np array
learn['inputs'] = np.array(learn['inputs'])
learn['targets'] = np.array(learn['targets'])

# load images for test
test = np.array(getITArray("mnist/testing", None)[0])

# start learning
print("learning form " + str(len(learn['inputs'])) + " samples")

netmodel.fit(
  rate=0.1,
  inputs=learn['inputs'],
  targets=learn['targets'],
  epochs=100,
  type='backPropagation'
)

print("try predict")
print("\n")
for sample in test:
  # print image
  print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
        for row in sample]))
  print("\n")
  # predict with image as inputs
  predict = netmodel.predict(sample)
  # print prediction
  print("it is " + str(np.argmax(predict)) + " with score " + str(np.max(predict)))
  print("\n\n")