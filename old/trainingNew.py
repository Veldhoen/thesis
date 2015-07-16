def setParameters(hyperParameters,nCores = 1, ada = True):
  global hyperParams
  hyperParams = hyperparameters

  global cores
  cores = nCores

  global adagrad
  adagrad = ada
  print 'Training hyperparameters are set.'

'''
   Epoch:
   - Divide the training data (examples['TRAIN']) into minibatches
   - Run each minibatch in parallel
   - Update theta (and the historical gradient) after each minibatch
   If earlyStopping is on:
   - Check for convergence using the training error and an estimate
     of the held-out error. If the expected improvement on generalization
     is too low, stop training immediately (i.e. do not finish the epoch)


'''
def epoch(theta, examples, historical_grad = None, earlyStopping = False):
  train = examples['TRAIN']
  trial = examples['TRIAL']
  if historical_grad is None: historical_grad = theta.zeros_like(False)
  if batchsiza<1: batchsize = len(data)

  mgr = Manager()
  random.shuffle(train)
  if earlyStopping: 
    random.shuffle(trial)
    trainStrip = Strip(hyperParams['stripL']
    trialStrip = Strip(hyperParams['stripL']


  for b in xrange((len(train)+batchsize-1)//batchsize):
    ns.theta = theta
    minibatch = data[batch*batchsize:(batch+1)*batchsize]
    s = (len(minibatch)+cores-1)//cores

    ps = []
    q = Queue()

    if cores<2: trainBatch(ns, minibatch,q) #don't start a subprocess
    else:
      for j in xrange(cores):
        p = Process(name='minibatch'+str(batch)+'-'+str(j), target=trainBatch, args=(ns, minibatch[j*s:(j+1)*s],q))
        ps.append(p)
        p.start()


    errors = []
    theta.regularize(hyperParams['alpha']/len(data), hyperParams['lambda'])
    for j in xrange(len(ps)):
      (grad, error) = q.get()
      if grad is None: continue
      if adagrad: theta.update(grad,hyperParams['alpha'],historical_grad)
      else: theta.update(grad,hyperParams['alpha'])
      errors.append(error)

    # make sure all worker processes have finished and are killed
    for p in trainPs: p.join()




    print '\tBatch:', batch, ', average error:', sum(errors)/len(errors), ', theta norm:', theta.norm()

'''Compute the average gradient over the batch of examples
input:
- ns: namespace containing theta
- examples: items in the (mini)batch
- q: Queue to put the result (gradient and error) in

For each example: have the network backpropagate errors and return a gradient

'''
def trainBatch(ns, examples, q):
  if len(examples)>0:
    theta = ns.theta
    lambdaL2 = ns.lamb
    grads = theta.zeros_like()
  #  regularization = lambdaL2/2 * theta.norm()**2
    error = 0
    for nw in examples:
      dgrads,derror = nw.train(theta)
      error+= derror
      for name in grads.keys():
        grads[name] = grads[name] + dgrads[name]/len(examples)
    q.put((grads, error/len(examples)))
  else:
    print '\tPart of minibatch with no training examples.'
    q.put((None,None))


class strip():
  def __init__(self, sz):
    self.size = sz
    self.trainL = []
    self.validL = []

  def append(self,trainE,validE):
    # fill the entire strip with the initial value
    if len(self.trainL) == 0:
      self.trainL = [trainE]*self.size
      self.validL = [validE]*self.size

    else: #pop the oldest values and append the new ones
      self.trainL.append(trainE)
      self.validL.append(validE)
      self.trainL=self.trainL[1:]
      self.validL=self.validL[1:]

  def average(self):
    if len(self.l)<1: return 0
    else: return sum(self.l)/len(self.l)
  
  # generalization loss
  # "the relative increase of the validation error over the minimum-so-far"
  def gl(self):
    self.validL[-1]/min(validL)-1
    
  # training progress
  # "how much was the average training error during the strip larger than the minimum training error during the strip?"
  def pk(self):
    sum(self.trainL)/(self.size*min(self.trainL))


  def pq(self):
    self.gl()/self.pk()
