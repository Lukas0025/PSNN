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

import copy, multiprocessing, pickle
from .loss.basic import mse as defaultloss

class model:
  '''
  create netmodel from layers array

  @param object self
  @param array of object layers - array of instances of layers classes - Optional
  @return None
  '''
  def __init__(self, layers = []):
     self.debug = False

     for layer in layers:
        if not(hasattr(layer, '__dict__')):
          raise ValueError('One of the layers is not class.')
     
     # store layers
     self.layers = layers[:]

  '''
  store model to file

  @param object self
  @param str file - file name
  @return None
  '''
  def dump(self, file):
     filehandler = open(file, "wb")
     pickle.dump(self.layers, filehandler)
     filehandler.close()

  '''
  store model to string

  @param object self
  @return pickle string
  '''
  def dumps(self):
     return pickle.dumps(self.layers)

  '''
  load model from file

  @param object self
  @param str file - file name
  @return None
  '''
  def load(self, file):
     filehandler = open(file, "rb")
     self.layers = pickle.load(filehandler)
     filehandler.close()

  '''
  load model from string

  @param object self
  @param str string - data from dumps()
  @return None
  '''
  def loads(self, string):
     self.layers = pickle.loads(string)

  '''
  add new layer to network model (when netmodel is not created)

  @param object self
  @param object layer - Instance of layer class
  @return None
  '''
  def add(self, layer):
     if not(hasattr(layer, '__dict__')):
        raise ValueError('layer is not class.')
     self.layers.append(layer)

  '''
  create netmodel init weigths and another thing
  using __create__() function in every layer

  @param object self
  @param tuple inputs - shape of input e.g (10,10)
  @return None
  '''
  def create(self, inputs):
     out = list(inputs)
     for layer in self.layers:
        out = layer.__create__(out)

  '''
  predict output form input data

  @param object self
  @param numpy array inputs - array of inputs for network
  @return numpy array - prediction
  '''
  def predict(self, inputs):
     out = inputs
     for layer in self.layers:
        out = layer.__forward__(out)
     return out

  '''
  clear memory based on Recurrnet architecture

  @param object self
  @return None
  '''
  def clrmem(self):
     for layer in self.layers:
        if hasattr(layer, '__clrmem__'):
          layer.__clrmem__()

  '''
  randomly mutate network using 
  random numbers based on rate. 
  Each layer can use the rate differently,
  usually generate random between -rate and rate

  @param object self
  @return None
  '''
  def mutate(self, rate):
     for layer in self.layers:
        if hasattr(layer, '__mutate__'):
          layer.__mutate__(rate)

  """
  create terminal progress bar

  @param int iteration - current iteration
  @param int total - total iterations
  @param str prefix - prefix string (default is '')
  @param str suffix - suffix string (default is '')
  @param int decimals - positive number of decimals in percent complete (default is 1)
  @param int length - character length of bar (default is 100)
  @param str fill - bar fill character (default is '█')
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

  '''
  do backpropagation for one inputdata and one taget

  @param object self
  @param numpy array inputdata - inputdata for network
  @param numpy array target - target (output) for inputdata
  @param object lossfunction - Instance of loss function class (default is MSE)
  @param float rate - rate value (size of correction jump)
  @return float - loss of model before learn
  '''
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

  '''
  do backpropagation for array of inputdata and array of tagets

  @param object self
  @param array of numpy arrays inputs - array of inputs data for network
  @param array of numpy array targets - array of targets (outputs) for inputs data
  @param object lossfunction - Instance of loss function class (default is MSE)
  @param function dynRateFunc - function to definite rate dynamic for every epoch (Default is None)
  @param float rate - rate value (size of correction jump)
  @param int epochs - number of backpropagation loops with data (default is 1)
  @param int offset - number of skiped elements for what do not do backpropagation only forward 
  @return float - loss of model before last learn loop
  '''
  def backPropagationFit(self, inputs, targets, lossfunc=None, dynRateFunc=None, rate=1, epochs=1, offset=0):
    
    if dynRateFunc != None:
      stockrate = rate

    for epoch in range(epochs):
      
      if dynRateFunc != None:
         rate = dynRateFunc(stockrate, epoch)
      
      loss = 0
      
      for i in range(len(inputs)):
        if offset <= i:
          loss += self.backPropagation(inputs[i], targets[i], lossfunc, rate)
        else:
           # do only forward no backprop
           self.predict(inputs[i])

        if self.debug:
           self.printProgressBar(
                                 i + 1,
                                 len(inputs),
                                 prefix = 'epoch ' + str(epoch + 1) + '/' + str(epochs),
                                 suffix = 'Complete AVG loss: ' + str(loss / (i + 1)),
                                 length = 20
          )

    return loss


  '''
  do learning using evolution (create copy of network, mutate every copy and select the copy closest to the target)

  @param object self
  @param array of numpy arrays inputs - array of inputs data for network (if is None lossfunc will be call as lossfunc(replication))
  @param array of numpy array targets - array of targets (outputs) for inputs data
  @param object lossfunction - Instance of loss function class (default is MSE)
  @param function dynRateFunc - function to definite rate dynamic for every epoch (Default is None)
  @param float rate - rate value for mutate() function (default is 1)
  @param int replication - number of copyes (default is 20)
  @param int epochs - number of backpropagation loops with data (default is 1)
  @param int offset - number of skiped elements for what do not do backpropagation only forward 
  @return float - loss of model before last learn loop
  '''
  def evolutionFit(self, inputs=None, targets=None, rate=1, replication=20, epochs=1, lossfunc=None, dynRateFunc=None, offset=0):
     if replication < 2:
        raise ValueError('min number of replications is 2')

     if dynRateFunc != None:
        stockrate = rate

     if lossfunc == None:
        lossfunc = defaultloss()

     for epoch in range(epochs):
        replications = []
        loss = []
        
        # use dyn Rate if is definite
        if dynRateFunc != None:
          rate = dynRateFunc(stockrate, epoch)

        # make copyes of network
        for i in range(replication):
          replications.append(copy.deepcopy(self))
          loss.append(0)

          # mutation
          if i != 0:
            replications[i].mutate(rate)
        
        # test every replication how is it best
        for j in range(replication):
          # clear Reccurent base memery
          replications[j].clrmem()
          # if inputs is not definite use loss function to get loss
          if inputs is None:
            loss[j] = lossfunc(replications[j])
          else:
            for i in range(len(inputs)):
              if offset <= i:
                # calc error
                loss[j] += lossfunc.__clac1V__(
                  replications[j].predict(inputs[i]),
                  targets[i]
                )
              else:
                 # do only forward no calc error
                 replications[j].predict(inputs[i])

              if self.debug:
                 self.printProgressBar(
                    j * len(inputs) + (i + 1),
                    replication * len(inputs),
                    prefix = 'epoch ' + str(epoch + 1) + '/' + str(epochs),
                    suffix = 'Complete AVG loss: ' + str(loss[0] / (i + 1)),
                    

        # select the best
        minLoss = 0

        for i in range(len(loss)):
          if loss[i] < loss[minLoss]:
            minLoss = i
         
        # set BEST as current
        self.layers = replications[minLoss].layers
     
     return loss[minLoss]

  '''
  fit model with data and targets with specific method

  @param object self
  @param str type - type of fit method evolution/backPropagation (default is "evolution")
  @param array of numpy arrays inputs - array of inputs data for network (if is None lossfunc will be call as lossfunc(replication))
  @param array of numpy array targets - array of targets (outputs) for inputs data
  @param object lossfunction - Instance of loss function class (default is MSE)
  @param function dynRateFunc - function to definite rate dynamic for every epoch (Default is None)
  @param float rate - rate value for  mutate() function (default is 1)
  @param int replication - number of copyes (default is 20)
  @param int epochs - number of backpropagation loops with data (default is 1)
  @param int offset - number of skiped elements for what do not do backpropagation only forward 
  @return float - loss of model before last learn loop
  '''
  def fit(self, inputs=None, targets=None, type="evolution", rate=1, replication=20, epochs=1, lossfunc=None, dynRateFunc=None, offset=0):
     if type == "evolution":
        return self.evolutionFit(inputs, targets, rate, replication, epochs, lossfunc, dynRateFunc, offset)
     elif type == "backPropagation":
        return self.backPropagationFit(inputs, targets, lossfunc, dynRateFunc, rate, epochs, offset)