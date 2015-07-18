import random
import pickle
from earlyStopping import stopNow
from multiprocessing import Process, Queue, Pool, Manager
import sys, os, pickle

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

def evaluateBit(theta, testData, q, sample=1):
  performance = [nw.evaluate(theta,sample) for nw in testData]
  q.put(sum(performance)/len(performance))

def evaluate(theta, testData, q = None, description = '', sample=1, cores=1):
  if cores>1:
    myQueue = Queue()
    pPs = []
    bitSize = len(testData)//cores+1
    for i in len(cores):
      databit = testData[i*bitSize:(i+1)*bitSize]
      p = Process(name='evaluate', target=evaluateBit, args=(theta, databit, myQueue,sample))
      pPs.append(p)
      p.start()

    performance = [myQueue.get() for p in len(pPs)]
  else: performance = [nw.evaluate(theta,sample) for nw in testData]

  if q is None:  return sum(performance)/len(performance)
  else:
    confusion = None
    q.put((description, sum(performance)/len(performance),confusion))





def phaseZero(tTreebank, vData, hyperParams, adagrad, theta, cores):
  if adagrad: histGrad = theta.gradient()
  else: histGrad = None
  histGrad.unSparse()
  print '\tStart training'

  trainLoss = []
  validLoss = []

  for i in range(10,40,2): # slowy increase sentence length
    examples = tTreebank.getExamples()
    tData = [e for e in examples if len(e.scoreNodes)<i]
    while len(tData)<len(examples):
      tData.extend([e for e in tTreebank.getExamples() if len(e.scoreNodes)<i])
    tData = tData[:len(examples)]

    print '\tIteration with sentences up to length',i,'('+str(len(tData))+' examples)'

    trainLoss.append(trainOnSet(hyperParams, tData, theta, adagrad, histGrad, cores))

    print '\tComputing performance...'
    performance = evaluate(theta, testData, q = None, description = '', sample=0.05, cores=cores) #[nw.evaluate(theta,0.05) for nw in vData]
    validLoss.append(sum(performance)/len(performance))

    print '\tTraining error:', trainLoss[-1], ', Estimated performance:', validLoss[-1]


def phase(tTreebank, vData, hyperParams, adagrad, theta, cores):
  if adagrad:
    histGrad = theta.gradient()
    histGrad.unSparse()
  else: histGrad = None

  print '\tStart training'

  trainLoss = []
  validLoss = []

  for i in xrange(hyperParams['nEpochs']):
    if stopNow(trainLoss, validLoss): break#converged/ overfitting:
    tData = tTreebank.getExamples()
    print '\tIteration',i,'('+str(len(tData))+' examples)'


    trainLoss.append(trainOnSet(hyperParams, tData, theta, adagrad, histGrad, cores))
    print '\tComputing performance...'
    performance = evaluate(theta, testData, q = None, description = '', sample=1, cores=cores)#[nw.evaluate(theta,0.05) for nw in vData]
    validLoss.append(sum(performance)/len(performance))
    print '\tTraining error:', trainLoss[-1], ', Estimated performance:', validLoss[-1]




def beginSmall(tTreebank, vTreebank, hyperParams, adagrad, theta, outDir, cores=1):
  vData = vTreebank.getExamples()

  qPerformance = Queue()
  pPs = []
  p = Process(name='evaluateINI', target=evaluate, args=(theta, vData, qPerformance,'Initial Performance on validation set:'))
  pPs.append(p)
  p.start()

  print 'Phase 0: no grammar specialization'
  phaseZero(tTreebank, vData, hyperParams, adagrad, theta, cores)
  # evaluate
  p = Process(name='evaluatePhase0', target=evaluate, args=(theta, vData, qPerformance,'Performance on validation set after phase 0:'))
  pPs.append(p)
  p.start()
  # store theta
  with open(os.path.join(outDir,'phase0.theta.pik'),'wb') as f:
    pickle.dump(theta,f)

  print 'Phase 1: head specialization'
  theta.specializeHeads()
  phase(tTreebank, vData, hyperParams, adagrad, theta, cores)

  # evaluate
  p = Process(name='evaluatePhase1', target=evaluate, args=(theta, vData, qPerformance,'Performance on validation set after phase 1:'))
  pPs.append(p)
  p.start()
  # store theta
  with open(os.path.join(outDir,'phase1.theta.pik'),'wb') as f:
    pickle.dump(theta,f)

  print 'Phase 2: rule specialization - most frequent', hyperParams['nRules']
  theta.specializeRules(hyperParams['nRules'])
  phase(tTreebank, vData, hyperParams, adagrad, theta, cores)

  # evaluate
  p = Process(name='evaluatePhase2', target=evaluate, args=(theta, vData, qPerformance,'Eventual performance on validation set after phase 2:'))
  pPs.append(p)
  p.start()
  # store theta
  with open(os.path.join(outDir,'phase2Final.theta.pik'),'wb') as f:
    pickle.dump(theta,f)

  # print results of evaluation
  for j in xrange(len(pPs)):
    description, accuracy, confusion = qPerformance.get()
    print description, accuracy
  # make sure all worker processes have finished and are killed
  for p in pPs: p.join()

def trainOnSet(hyperParams, examples, theta, adagrad, histGrad, cores):
  mgr = Manager()
  ns = mgr.Namespace()
  ns.lamb = hyperParams['lambda']
  batchsize = hyperParams['bSize']
  random.shuffle(examples) # randomly split the data into parts of batchsize
  avErrors = []
  for batch in xrange((len(examples)+batchsize-1)//batchsize):
    ns.theta = theta
    minibatch = examples[batch*batchsize:(batch+1)*batchsize]
    s = (len(minibatch)+cores-1)//cores
    trainPs = []
    q = Queue()

    if cores<2:
      trainBatch(ns, minibatch,q) #don't start a subprocess
      trainPs.append('')  # But do put a placeholder in the queue
    else:
      for j in xrange(cores):
        p = Process(name='minibatch'+str(batch)+'-'+str(j), target=trainBatch, args=(ns, minibatch[j*s:(j+1)*s],q))
        trainPs.append(p)
        p.start()


    errors = []
    theta.regularize(hyperParams['alpha']/len(examples), hyperParams['lambda'])
    for j in xrange(len(trainPs)):
      (grad, error) = q.get()
      if grad is None: continue
      if adagrad: theta.update(grad,hyperParams['alpha'],histGrad)
      else: theta.update(grad,hyperParams['alpha'])
      errors.append(error)

    # make sure all worker processes have finished and are killed
    if cores>1:
      for p in trainPs: p.join()

    try: avError = sum(errors)/len(errors)
    except:
      avError = 0
      print 'batch size zero!'
    if True: #batch % 10 == 0:
      print '\t\tBatch', batch, ', average error:',avError , ', theta norm:', theta.norm()
    avErrors.append(avError)
  return sum(avErrors)/len(avErrors)


def trainBatch(ns, examples, q=None):
  if len(examples)>0:
    theta = ns.theta
    lambdaL2 = ns.lamb
    grads = theta.gradient()
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