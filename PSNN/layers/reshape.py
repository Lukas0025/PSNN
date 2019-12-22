#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## @package PSNN.layers.reshape
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# reshape layer

import numpy as np


## reshape layer
# 
# input ND numpy arrays
# output ND numpy array with shape (num of elements) -> with specific shape
# 
# SAMPLE:
# input-shape: (10, 2)
# reshape((-1,10))
# output-shape(2, 10)
class reshape:
    ##
    # @param self object
    # @param outshape shape what you want on output
    def __init__(self, outshape):
        self.outshape = outshape

    ##
    # @param self object
    # @param inputshape 1D array of int inputshape
    def __create__(self, inputshape):
        #create test data with shape
        testdata = np.zeros(inputshape).reshape(self.outshape)

        return list(testdata.shape)

    ##
    # @param self object
    # @param inputs ND array of float inputs
    def __forward__(self, inputs):
        return inputs.reshape(self.outshape)