import random
import pickle
from multiprocessing import Process, Queue, Pool, Manager

class Treebank():
  def __init__(self,fileList):
    self.files = fileList
    self.reset()

  def getExamples(self):
    if len(self.files) == 1: aFile = self.files[0]
    else:
      try: aFile = self.it.next()
      except:
        self.reset()
        aFile = self.it.next()
    with open(aFile,'rb') as f:
      examples = pickle.load(f)
    return examples

  def addFiles(self,fileList):
    self.files.extend(fileList)
    self.reset()
  def reset(self):
    random.shuffle(self.files)
    self.it = iter(self.files)

def evaluateQueue(theta, testData, q = None, description = '', sample=1):
  performance = [nw.evaluate(theta,sample) for nw in testData]
  confusion = None
  q.put((description, sum(performance)/len(performance),confusion))

def phase(tTreebank, vData, hyperParams, theta, cores):
  histGrad = theta.gradient()
  histGrad.unSparse()
  print '\tStart training'

  trainLoss = []
  validLoss = []
#  while not converged/ overfitting:
  for i in range(1):
    tData = tTreebank.getExamples()
    print '\tIteration',i,'('+str(len(tData))+' examples)'


    trainLoss.extend(trainOnSet(hyperParams, tData, theta, histGrad, cores))
    performance = [nw.evaluate(theta,0.05) for nw in vData]
    validLoss.extend(sum(performance)/len(performance))

    print '\tTraining error:', trainLoss[-1], 'Estimated performance:', validLoss[-1]




def beginSmall(tTreebank, vTreebank, hyperParams, theta, cores=1):
  vData = vTreebank.getExamples()
  
  qPerformance = Queue()
  pPs = []
  p = Process(name='evaluateINI', target=evaluateQueue, args=(theta, vData, qPerformance,'Initial Performance on validation set:'))
  pPs.append(p)
  p.start()

  print 'Phase 1: no grammar specialization'
  phase(tTreebank, vData, hyperParams, theta, cores)

  p = Process(name='evaluatePhase1', target=evaluateQueue, args=(theta, vData, qPerformance,'Performance on validation set after phase 1:'))
  pPs.append(p)
  p.start()

  print 'Phase 2: head specialization'
  theta.specializeHeads()
  phase(tTreebank, vData, hyperParams, theta, cores)

  p = Process(name='evaluatePhase1', target=evaluateQueue, args=(theta, vData, qPerformance,'Performance on validation set after phase 2:'))
  pPs.append(p)
  p.start()

  print 'Phase 3: rule specialization - most frequent', hyperParams['nRules']
  theta.specializeRules(hyperParams['nRules'])
  phase(tTreebank, vData, hyperParams, theta, cores)

  p = Process(name='evaluatePhase1', target=evaluateQueue, args=(theta, vData, qPerformance,'Eventual performance on validation set after phase 3:'))
  pPs.append(p)
  p.start()


  for j in xrange(len(pPs)):
    description, (accuracy, confusion) = qPerformance.get()
    print description, accuracy
  # make sure all worker processes have finished and are killed
  for p in pPs: p.join()

def trainOnSet(hyperParams, examples, theta, histGrad, cores):

  mgr = Manager()
  ns = mgr.Namespace()
  ns.lamb = hyperParams['lambda']
  random.shuffle(examples) # randomly split the data into parts of batchsize
  avErrors = []
  for batch in xrange((len(examples)+batchsize-1)//batchsize):
    ns.theta = theta
    minibatch = examples[batch*batchsize:(batch+1)*batchsize]
#      print 'minibatch size:',len(minibatch)
#      minibatch = random.sample(data, batchsize)
    s = (len(minibatch)+cores-1)//cores
    trainPs = []
    q = Queue()

    if cores<2: trainBatch(ns, minibatch,q) #don't start a subprocess
    else:
      for j in xrange(cores):
        p = Process(name='minibatch'+str(batch)+'-'+str(j), target=trainBatch, args=(ns, minibatch[j*s:(j+1)*s],q))
        trainPs.append(p)
        p.start()


    errors = []
    theta.regularize(hyperParams['alpha']/len(data), hyperParams['lambda'])
    for j in xrange(len(trainPs)):
      (grad, error) = q.get()
      if grad is None: continue
      if adagrad: theta.update(grad,hyperParams['alpha'],histGrad)
      else: theta.update(grad,hyperParams['alpha'])
      errors.append(error)

    # make sure all worker processes have finished and are killed
    for p in trainPs: p.join()


    if True: #batch % 10 == 0:
      print '\t\tBatch', batch, ', average error:', sum(errors)/len(errors), ', theta norm:', theta.norm()
    avErrors.append(sum(errors)/len(errors))
  return sum(avErrors)/len(avErrors)


def trainBatch(ns, examples, q=None):
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