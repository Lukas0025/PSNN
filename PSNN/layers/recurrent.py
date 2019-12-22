#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## @package PSNN.layers.recurrent
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# recurrent layer

import numpy as np


## recurrent layer
#
# input ND numpy arrays
# output ND numpy array with shape (MEM SIZE, ...all input shape numbers)
#
# SAMPLE:
# input-shape: (10, 10, 2)
# recurrent(3)
# output-shape(3, 10, 10, 2) 
class recurrent:
    ##
    # @param self object
    # @param size size of recurrent memory
    # @return None
    def __init__(self, size):
        self.size = size

    ## Clear memory
    # @param self object
    # @return None
    def __clrmem__(self):
        self.mem = []

        for i in range(self.size):
            self.mem.append(np.zeros(self.shape))

    ## Add to memory (Remember) and delete oldest
    # @param self object
    # @param data data to store
    # @return None
    def addToMem(self, data):
        del self.mem[-1]
        self.mem.insert(0, data)

    ##
    # @param self object
    # @param inputshape 1D array of int
    # @return output shape
    def __create__(self, inputshape):
        self.shape = inputshape[:]
        self.__clrmem__()

        inputshape.insert(0, self.size)
        return inputshape

    ## do forward
    # @param self object
    # @param inputs ND array of float inputs
    # @return forwarded inputs
    def __forward__(self, inputs):
        self.addToMem(inputs)
        return np.array(self.mem)