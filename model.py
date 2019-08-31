#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  model.py
#  
#  Copyright 2019 Lukáš Plevač <lukasplevac@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#

import copy, threading

class model:
	'''
	@param object self
    @param array of classes (layers) layers
	'''
	def __init__(self, layers = []):
		self.debug = False

		for layer in layers:
			if not(hasattr(layer, '__dict__')):
				raise ValueError('One of the layers is not class.')
		
		# store layers
		self.layers = layers[:]

	'''
	@param object self
    @param class (layer) layer
	'''
	def add(self, layer):
		if not(hasattr(layer, '__dict__')):
			raise ValueError('layer is not class.')
		self.layers.append(layer)

	'''
	@param object self
    @param tuple of shape inputs
	'''
	def create(self, inputs):
		out = list(inputs)
		for layer in self.layers:
			out = layer.__create__(out)

	'''
	@param object self
    @param numpy array inputs
	'''
	def predict(self, inputs):
		out = inputs
		for layer in self.layers:
			out = layer.__forward__(out)
		return out

	def evolute(self, rate):
		for layer in self.layers:
			layer.__evolute__(rate)

	"""
	Call in a loop to create terminal progress bar
	@param iteration   - Required  : current iteration (Int)
	@param total       - Required  : total iterations (Int)
	@param prefix      - Optional  : prefix string (Str)
	@param suffix      - Optional  : suffix string (Str)
	@param decimals    - Optional  : positive number of decimals in percent complete (Int)
	@param length      - Optional  : character length of bar (Int)
	@param fill        - Optional  : bar fill character (Str)
	"""
	def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
		# Print New Line on Complete
		if iteration == total: 
			print()
			print()

	def evolutionFit(self, inputs=None, targets=None, rate=1, replication=20, epochs=1, lossfunc=None, dynRateFunc=None):
		if lossfunc == None:
			raise ValueError('lossfunc cant by None')

		if replication < 2:
			raise ValueError('min number of replications is 2')

		if dynRateFunc != None:
			stockrate = rate

		for epoch in range(epochs):
			replications = []
			loss = []

			if dynRateFunc != None:
				rate = dynRateFunc(stockrate, epoch)

			for i in range(replication):
				replications.append(copy.deepcopy(self))
				loss.append(0)

				#evolute
				if i != 0:
					replications[i].evolute(rate)
			
			for j in range(replication):
				if inputs is None:
					loss[j] = lossfunc(
						replications[j]
					)

					if self.debug:
						self.printProgressBar(
							j + 1,
							replication,
							prefix = 'epoch ' + str(epoch + 1) + '/' + str(epochs),
							suffix = 'Complete AVG loss: ' + str(loss[0] / (i + 1)),
							length = 20
						)
				else:
					for i in range(len(inputs)):
						if targets is None:
							loss[j] += lossfunc(
								replications[j],
								inputs[i]
							)
						else:	
							loss[j] += lossfunc(
								replications[j].predict(inputs[i]),
								targets[i]
							)

						if self.debug:
							self.printProgressBar(
								j * len(inputs) + (i + 1),
								replication * len(inputs),
								prefix = 'epoch ' + str(epoch + 1) + '/' + str(epochs),
								suffix = 'Complete AVG loss: ' + str(loss[0] / (i + 1)),
								length = 20
							)

			#select the best
			minLoss = 0

			for i in range(len(loss)):
				if loss[i] < loss[minLoss]:
					minLoss = i
				
			self.layers = replications[minLoss].layers

	def fit(self, inputs=None, targets=None, type="evolution", rate=1, replication=20, epochs=1, lossfunc=None, dynRateFunc=None):
		if type == "evolution":
			return self.evolutionFit(inputs, targets, rate, replication, epochs, lossfunc, dynRateFunc)