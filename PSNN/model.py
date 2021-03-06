#!/usr/bin/env python	
# -*- coding: utf-8 -*-	
## @package PSNN.model
#  @author Lukáš Plevač <lukasplevac@gmail.com>
#  @date 22.12.2019
#  
# model controller
# main class is PSNN.model.model (can be called as PSNN.model)
# another is for interal use

import copy, multiprocessing, pickle, requests, base64
from .loss.basic import mse as defaultloss

class model:
   ## create netmodel from layers array
   #
   # @param self object
   # @param layers array of object layers - array of instances of layers classes - Optional
   # @return None
   def __init__(self, layers = []):
      ## debug varaible
      # default set to 0
      # for debug set 1, 2, 3
      self.debug = 0
      
      ## Size of multiprocessing pool (number of theards)
      self.poolSize = 4

      ## model server address
      # default set to public models server
      self.modelsServer = "http://pysnn.jecool.net/api.php"

      for layer in layers:
         if not(hasattr(layer, '__dict__')):
            raise ValueError('One of the layers is not class.')
     
      # store layers
      self.layers = layers[:]

   ## store model to file
   #
   # @param self object
   # @param file str file name
   # @return None
   def dump(self, file):
      filehandler = open(file, "wb")
      pickle.dump(self.layers, filehandler)
      filehandler.close()

   ## store model to string
   #
   # @param self object
   # @return pickle string
   def dumps(self):
      obj = pickle.dumps(self.layers)
      return base64.b64encode(obj)

   ## load model from file
   # 
   # @param self object
   # @param file - file name
   # @return None
   def load(self, file):
      filehandler = open(file, "rb")
      self.layers = pickle.load(filehandler)
      filehandler.close()

   ## load model from string
   #
   # @param self object
   # @param string - data from dumps()
   # @return None
   def loads(self, string):
      obj = base64.b64decode(string)
      self.layers = pickle.loads(obj)

   ## get model from models server
   #
   # @param self object
   # @param modelID - id of model on server
   # @return None
   def get(self, modelID):
      data = {
         'a': 'getModel',
         'name': modelID
      }

      r = requests.post(
         url = self.modelsServer,
         data = data
      )

     self.loads(r.text)

  
   ## add new layer to network model (when netmodel is not created)
   #
   # @param self object
   # @param layer - Instance of layer class
   # @return None
   def add(self, layer):
      if not(hasattr(layer, '__dict__')):
         raise ValueError('layer is not class.')
      self.layers.append(layer)

   ## create netmodel init weigths and another thing
   # using __create__() function in every layer
   #
   # @param self object
   # @param inputs - shape of input e.g (10,10)
   # @return None
   def create(self, inputs):
      out = list(inputs)
      for layer in self.layers:
         out = layer.__create__(out)

   ## predict output form input data
   #
   # @param self object
   # @param inputs - array of inputs for network
   # @return numpy array - prediction
   def predict(self, inputs):
      out = inputs
      for layer in self.layers:
         out = layer.__forward__(out)
      return out

   ## clear memory based on Recurrnet architecture
   #
   # @param self object
   # @return None
   def clrmem(self):
      for layer in self.layers:
         if hasattr(layer, '__clrmem__'):
            layer.__clrmem__()

  
   ## randomly mutate network using 
   # random numbers based on rate. 
   # Each layer can use the rate differently,
   # usually generate random between -rate and rate
   #
   # @param self object
   # @param rate rate of mutation
   # @param layerNum (optimal) number of layer to mutate
   # @return None
   def mutate(self, rate, layerNum = None):
      if not(layerNum is None):
         if hasattr(self.layers[layerNum], '__mutate__'):
            self.layers[layerNum].__mutate__(rate)
      else:
         for layer in self.layers:
            if hasattr(layer, '__mutate__'):
               layer.__mutate__(rate)


  
   ## do backpropagation for one inputdata and one taget
   #
   # @param self object
   # @param inputdata inputdata for network
   # @param target target (output) for inputdata
   # @param lossfunc Instance of loss function class (default is MSE)
   # @param rate rate value (size of correction jump)
   # @return loss of model before learn
   def backPropagation(self, inputdata, target, lossfunc=None, rate=1):
      if lossfunc == None:
         lossfunc = defaultloss()

      inputs = []
      inputs.append(inputdata)
    
      # do predict first
      for layer in self.layers:
         inputs.append(layer.__forward__(inputs[-1]))
	  
      # calc error on output
      fail = lossfunc.__calc__(inputs[-1], target)
        
      for i in range(len(self.layers) - 1, -1, -1):
         if hasattr(self.layers[i], '__backprop__'):
            fail = self.layers[i].__backprop__(inputs[i], inputs[i + 1], fail, rate)
		
      return lossfunc.__clac1V__(inputs[-1], target)

   ## do backpropagation for array of inputdata and array of tagets
   #
   # @param self object
   # @param inputs array of inputs data for network
   # @param targets array of targets (outputs) for inputs data
   # @param lossfunc Instance of loss function class (default is MSE)
   # @param dynRateFunc function to definite rate dynamic for every epoch (Default is None)
   # @param rate rate value (size of correction jump)
   # @param epochs number of backpropagation loops with data (default is 1)
   # @param offset number of skiped elements for what do not do backpropagation only forward 
   # @return loss of model before last learn loop
   def backPropagationFit(self, inputs, targets, lossfunc=None, dynRateFunc=None, rate=1, epochs=1, offset=0):
    
      if dynRateFunc != None:
         stockrate = rate

      for epoch in range(epochs):
         counter = MPCounter(0, len(inputs))
      
      loss = 0
      
      for i in range(len(inputs)):
         if offset <= i:
            loss += self.backPropagation(inputs[i], targets[i], lossfunc, rate)
         else:
            # do only forward no backprop
            self.predict(inputs[i])

         if self.debug >= 2:
            counter.increment()
            counter.printProgressBar(prefix = 'epoch {} / {}'.format(epoch + 1, epochs))

      if dynRateFunc != None:
         rate = dynRateFunc(stockrate, epoch, loss)

      if self.debug >= 1:
         print("[INFO] actual epoch is {} loss is {}".format(epoch + 1, loss))
    
      return loss


   ## do learning using evolution (create copy of network, mutate every copy and select the copy closest to the target)
   #
   # @param self object
   # @param inputs array of inputs data for network (if is None lossfunc will be call as lossfunc(replication))
   # @param targets array of targets (outputs) for inputs data
   # @param lossfunc Instance of loss function class (default is MSE)
   # @param dynRateFunc function to definite rate dynamic for every epoch (Default is None)
   # @param rate rate value for mutate() function (default is 1)
   # @param replication number of copyes (default is 20)
   # @param epochs number of backpropagation loops with data (default is 1)
   # @param offset number of skiped elements for what do not do backpropagation only forward 
   # @param layer number of layer to fit (optimal) else fit all layers at same time
   # @return loss of model before last learn loop
   def evolutionFit(self, inputs=None, targets=None, rate=1, replication=20, epochs=1, lossfunc=None, dynRateFunc=None, offset=0, layer=None):

      if replication < 2:
         raise ValueError('min number of replications is 2')

      if dynRateFunc != None:
         stockrate = rate

      if lossfunc == None:
         lossfunc = defaultloss()

      for epoch in range(epochs):
         replications = []
         loss = []

         # make copyes of network
         for i in range(replication):
            replications.append(copy.deepcopy(self))
            loss.append(0)

            # mutation
            if i != 0:
               replications[i].mutate(rate, layer)


         # test every replication how is it best
         loss = []
         p = multiprocessing.Pool(self.poolSize)
         
         lossCalc = instanceLoss(
            inputs = inputs,
            targets = targets,
            lossfunc = lossfunc,
            offset = offset,
            toComplete = 0 if inputs is None else (len(inputs) * replication),
            epoch = epoch,
            epochs = epochs
         )

         loss = p.map(lossCalc.__calc__, replications)
         p.close()
         p.join()

         # select the best
         minLoss = 0

         for i in range(len(loss)):
            if loss[i] < loss[minLoss]:
               minLoss = i
         
         # set BEST as current
         self.layers = replications[minLoss].layers

         # use dyn Rate if is definite
         if dynRateFunc != None:
            rate = dynRateFunc(stockrate, epoch, loss[minLoss])

         if self.debug >= 1:
            print("[INFO] actual epoch is {} loss is {}".format(epoch + 1, loss[minLoss]))
     
      return loss[minLoss]

   ## fit model with data and targets with specific method
   #
   # @param self object
   # @param type type of fit method evolution/backPropagation/layerEvolution (default is "evolution")
   # @param inputs array of inputs data for network (if is None lossfunc will be call as lossfunc(replication))
   # @param targets array of targets (outputs) for inputs data
   # @param lossfunc Instance of loss function class (default is MSE)
   # @param dynRateFunc function to definite rate dynamic for every epoch (Default is None)
   # @param rate rate value for  mutate() function (default is 1)
   # @param replication number of copyes (default is 20)
   # @param epochs number of backpropagation loops with data (default is 1)
   # @param offset number of skiped elements for what do not do backpropagation only forward 
   # @return loss of model before last learn loop
   def fit(self, inputs=None, targets=None, type="evolution", rate=1, replication=20, epochs=1, lossfunc=None, dynRateFunc=None, offset=0):
      if type == "evolution":
         return self.evolutionFit(inputs, targets, rate, replication, epochs, lossfunc, dynRateFunc, offset)
      elif type == "backPropagation":
         return self.backPropagationFit(inputs, targets, lossfunc, dynRateFunc, rate, epochs, offset)
      elif type == "layerEvolution":
         loss = 0
        
         for layerNum in range(len(self.layers)):
            if self.debug >= 1:
               print("[INFO] fit layer with number {}".format(layerNum))
           
            if hasattr(self.layers[layerNum], '__mutate__'):
               loss = self.evolutionFit(inputs, targets, rate, replication, epochs, lossfunc, dynRateFunc, offset, layerNum)
        
      return loss

##
# this class is for multiprocess finished counter
class MPCounter(object):
   
   ## init function
   #
   # @param self object
   # @param initval start counter with number (default is 0)
   # @param total int number max for this count (when job is complete)
   # @return None
   def __init__(self, initval = 0, total = 1):
      manager = multiprocessing.Manager()
      self.val = manager.Value('i', initval)
      self.lock = manager.Lock()
      self.total = total

   ## increment counter (+1)
   #
   # @param self object
   # @return None
   def increment(self):
      with self.lock:
         self.val.value += 1

   ## get counter value
   #
   # @param self object
   # @return int value of counter
   def value(self):
      with self.lock:
         return self.val.value

   ## print progress bar by counter value
   #
   # @param self object
   # @param prefix text before bar (default is '')
   # @param length length of progress bar
   # @param fill complete fill with symbol
   # @return int value of counter
   def printProgressBar(self, prefix = '', length = 35, fill = '█'):
      with self.lock:
         percent = ("{0:.1f}").format(100 * (self.val.value / float(self.total)))
         filledLength = int(length * self.val.value // self.total)
         bar = fill * filledLength + '-' * (length - filledLength)
         print('\r%s |%s| %s%%' % (prefix, bar, percent), end = '\r')
         # Print New Line on Complete
         if self.val.value == self.total:
            self.val.value += 1
            print()

## 
# this class is for multiprocess loss calculing
class instanceLoss:
   ## init function
   #
   # @param self object
   # @param inputs array of inputs data for network (if is None lossfunc will be call as lossfunc(replication))
   # @param targets array of targets (outputs) for inputs data
   # @param lossfunc Instance of loss function class
   # @param replication number of copyes
   # @param epoch number of actual epoch for debug
   # @param epochs number of backpropagation loops with data (default is 1)
   # @param offset number of skiped elements for what do not do backpropagation only forward
   # @param toComplete total nuber of testing data (len(inputs) * num of replications)
   # @return None
   def __init__(self, inputs, targets, lossfunc, offset = 0, epoch = 1, epochs = 1, toComplete = 1):
      self.counter = MPCounter(0, toComplete)
      self.epoch = epoch
      self.epochs = epochs
      self.inputs = inputs
      self.targets = targets
      self.lossfunc = lossfunc
      self.offset = offset

   ## this function calc loss of instance of network model
   #
   # @param self object
   # @param instance instace of network model (model class)
   # @return Float loss of instance
   def __calc__(self, instance):
     # clear Reccurent base memery
     instance.clrmem()
     # if inputs is not definite use loss function to get loss
     if self.inputs is None:
       loss = self.lossfunc(instance)
     else:
       loss = 0
       for i in range(len(self.inputs)):
         if self.offset <= i:
           # calc error
           loss += self.lossfunc.__clac1V__(
             instance.predict(self.inputs[i]),
             self.targets[i]
           )
         else:
            # do only forward no calc error
            instance.predict(self.inputs[i])

         if instance.debug >= 2:
            self.counter.increment()
            self.counter.printProgressBar(prefix = 'epoch {} / {}'.format(self.epoch + 1, self.epochs))

 
     return loss
