from __future__ import division
import numpy as np
import random
import IORNN
from collections import defaultdict, Counter
import pickle, os
import time

from multiprocessing import Process, Queue

def thetaNorm(theta):
  names = theta.dtype.names
  return sum([np.linalg.norm(theta[name]) for name in names])/len(names)

def evaluate(theta, testData, amount=1):
  if isinstance(testData[0], IORNN.Node):
    return NAR(theta,testData, amount),None
  else: return accuracy(theta,testData)

def evaluateQueue(theta, testData, amount=1, q = None):
  if isinstance(testData[0], IORNN.Node):
    q.put((NAR(theta,testData, amount),None))
  else: q.put(accuracy(theta,testData))

def accuracy(theta, testData):
  true = 0
  confusion = defaultdict(Counter)
  for nw, tar in testData:
    pred = nw.predict(theta)
    confusion[tar][pred] += 1
    if pred == tar: true +=1
  return true/len(testData), confusion



# compute normalized average rank:
# for each leaf, compute scores for vocabulary, determine rank of actual word
# take the average of those ranks and normalize with vocabulary length
def NAR(theta,testData, amount=1):
  if amount<= 1: n = int(amount*len(testData))
  else: n = amount
  ranks = 0
  num = 0
  nwords = len(theta['wordIM'])
  for nw in random.sample(testData,n): # take a random sample of the test data (for efficiency. Does this make sense?)
    nw.activateNW(theta)
    leaves = nw.leaves()
    # we don't expect the network to make true predictions
    # for such small trees
    if len(leaves)<5: continue

    for leaf in random.sample(leaves,5): # take a random sample of the leaves (for efficiency. Does this make sense?)
      scores = [leaf.score(theta,x,False)[0] for x in xrange(nwords)]
      ranking = np.array(scores).argsort()[::-1].argsort()
      ranks+= ranking[leaf.index]
      num +=1
  return ranks/(nwords*num) #average (/num) and normalize (/nwords) the ranks

def confusionString(confusion, relations):
  if confusion is None: return ""
  st = '\t'+'\t'.join(relations)
  for tar in xrange(len(relations)):
    st+='\n'+relations[tar]
    for pred in xrange(len(relations)):
      st+='\t'+str(confusion[tar][pred])
    tot = sum(confusion[tar].values())
    st+= '\t'+str(tot)
    if tot>0: st+= '\t'+str(round(confusion[tar][tar]/tot,3))
    else: st+= '\t0'
  st+='\n'
  next = '\n'
  for pred in xrange(len(relations)):
    tot = sum([confusion[tar][pred] for tar in xrange(len(relations))])
    st += '\t'+str(tot)
    if tot > 0: next+= '\t'+str(round(confusion[pred][pred]/tot,3))
    else: next+= '\t0'
  st+= next
  return st

# although the numpy element-wise math operations are more
# efficient than iterating over the elements, they caused
# memory issues, especially in the case of the word matrix
# which has only few non-zero elements

def updateTheta(theta, gradient,histGradient,alpha):
  for name in theta.dtype.names:
    grad = gradient[name]
    nz = np.nonzero(grad)
    if len(nz[0]) == 0: continue
    histGrad = histGradient[name]
    for i in np.nditer(nz):
      histGrad[i] += np.square(grad[i])
      theta[name][i] += alpha * grad[i]/(np.sqrt(histGrad[i])+1e-6) # should be += rather than -=, right?!



# each minibatch is an independent random sample (without replacement)
def SGD(theta, hyperParams, examples, relations, cores = 1):
  testSample = 0.1
  data = examples['TRAIN']
  accuracy = 0.5
#  accuracy, confusion = evaluate(theta,examples['TEST'],testSample)

  print 'Start SGD training with random minibatches'
  historical_grad = np.zeros_like(theta)
  if hyperParams['bSize']: batchsize =hyperParams['bSize']
  else: batchsize = len(data)
#  while not converged:
  qPerformance = Queue()
  for i in xrange(hyperParams['nEpochs']):
    print 'Iteration', i #,', Performance on test sample:', accuracy
    random.shuffle(data) # randomly split the data into parts of batchsize
    for batch in xrange(len(data)//batchsize):
      minibatch = data[batch*batchsize:(batch+1)*batchsize]
#      minibatch = random.sample(data, batchsize)
      s = len(minibatch)//cores
#      processes = []
      q = Queue()
      for i in xrange(cores):
        p = Process(name='process'+str(i), target=epoch, args=(theta, minibatch[i*s:(i+1)*s], hyperParams['lambda'],q))
#        processes.append(p)
        p.start()

      errors = []
      for i in xrange(cores):
        (grad, error) = q.get()
        updateTheta(theta, grad,historical_grad,hyperParams['alpha'])
        errors.append(error)




#      grad, error = epoch(theta, minibatch, lambdaL2)
#      updateTheta(theta, grad,historical_grad,alpha)
      if batch % 10 == 0:
        print '\tBatch', batch, ', average error:', sum(errors)/len(errors), ', theta norm:', thetaNorm(theta)
    accuracy, confusion = evaluateQueue(theta,examples['TEST'],testSample, qPerformance)
  print 'Training terminated.'
  i = 0
  while i<nEpochs:
    accuracy, conf = qPerformance.get()
    print 'Iteration', i, ', Performance on test sample:', accuracy
    i+= 0
  accuracy, confusion = evaluate(theta,examples['TEST'])
  print 'Eventual performance on entire test set:', accuracy
  print confusionString(confusion, relations)
  with open(os.path.join('models','flickrIO.pik'), 'wb') as f:
    pickle.dump(theta, f, -1)


def epoch(theta, examples, lambdaL2, q=None):
  grads = np.zeros_like(theta)
  regularization = lambdaL2/2 * thetaNorm(theta)**2
  error = 0
  for nw in examples:
    dgrads,derror = nw.train(theta)
    error += derror
    for name in grads.dtype.names:
      grads[name] += dgrads[name]

  for name in grads.dtype.names:
    grads[name] = grads[name]/len(examples)+ lambdaL2*theta[name] # regularize
  q.put((grads, error/len(examples)))