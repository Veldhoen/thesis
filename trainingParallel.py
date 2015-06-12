from __future__ import division
import numpy as np
import random
import IORNN
from collections import defaultdict, Counter
import pickle, os
import time

from multiprocessing import Process, Queue, Pool, Manager

# def thetaNorm(theta):
#   names = theta.dtype.names
#   return sum([np.linalg.norm(theta[name]) for name in names])/len(names)

def evaluate(theta, testData, amount=1):
  if isinstance(testData[0], IORNN.Node):
    return NAR(theta,testData, amount),None
  else: return accuracy(theta,testData)

def evaluateQueue(theta, testData, q = None, description = '', amount=1):
  if isinstance(testData[0], IORNN.Node):
    nar = NAR(theta,testData, amount)
    q.put((description, (nar,None)))
  else: q.put((description, accuracy(theta,testData)))

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

    for leaf in leaves:
    #random.sample(leaves,5): # take a random sample of the leaves (for efficiency. Does this make sense?)
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

def SGD(theta, hyperParams, examples, relations, cores = 1, adagrad = True):
  data = examples['TRAIN']
  nEpochs = hyperParams['nEpochs']
#  accuracy = 0.5
  print 'Start SGD training'
  if adagrad: historical_grad = theta.zeros_like(False)
  if hyperParams['bSize']: batchsize =hyperParams['bSize']
  else: batchsize = len(data)
#  while not converged:
  qPerformance = Queue()
  pPs = []

  if cores<2: #don't start a subprocess
      evaluateQueue(theta, examples['TRIAL'], qPerformance,'Epoch '+ str(i)+', Performance on validation set:')
      while not qPerformance.empty():
        description, (accuracy, confusion) = qPerformance.get()
        print description, accuracy
    else:
      p = Process(name='evaluateINI', target=evaluateQueue, args=(theta, examples['TRIAL'], qPerformance,'Initial Performance on validation set:'))
      pPs.append(p)
      p.start()


  for i in xrange(nEpochs):
    print 'Epoch', i #,', Performance on test sample:', accuracy

    mgr = Manager()
    ns = mgr.Namespace()
    ns.lamb = hyperParams['lambda']
    random.shuffle(data) # randomly split the data into parts of batchsize
    for batch in xrange((len(data)+batchsize-1)//batchsize):
      ns.theta = theta
      minibatch = data[batch*batchsize:(batch+1)*batchsize]
      print 'minibatch size:',len(minibatch)
#      minibatch = random.sample(data, batchsize)
      s = (len(minibatch)+cores-1)//cores
      trainPs = []
      q = Queue()

      if cores<2: trainBatch(ns, minibatch,q) #don't start a subprocess
      else:
        for j in xrange(cores):
          p = Process(name='epoch'+str(i)+'minibatch'+str(batch)+'-'+str(j), target=trainBatch, args=(ns, minibatch[j*s:(j+1)*s],q))
          trainPs.append(p)
          p.start()

        # wait or all worker processes to finish
        for p in trainPs: p.join()

      errors = []
      theta.regularize(hyperParams['alpha'], hyperParams['lambda'], len(data))
      for j in xrange(cores):
        (grad, error) = q.get()
        if grad is None: continue
        if adagrad: theta.update(grad,hyperParams['alpha'],historical_grad)
        else: theta.update(grad,hyperParams['alpha'])
        errors.append(error)

      if batch % 10 == 0:
        print '\tBatch', batch, ', average error:', sum(errors)/len(errors), ', theta norm:', theta.norm()

    if cores<2: trainBatch(ns, minibatch,q) #don't start a subprocess
      evaluateQueue(theta, examples['TRIAL'], qPerformance,'Epoch '+ str(i)+', Performance on validation set:')
      while not qPerformance.empty():
        description, (accuracy, confusion) = qPerformance.get()
        print description, accuracy
    else:
      p = Process(name='evaluate'+str(i), target=evaluateQueue, args=(theta, examples['TRIAL'], qPerformance,'Epoch '+ str(i)+', Performance on validation set:'))
      pPs.append(p)
      p.start()

  print 'Computing performance...',len(pPs)

  if cores<2: trainBatch(ns, minibatch,q) #don't start a subprocess
    evaluateQueue(theta, examples['TEST'], qPerformance,'Eventual performance on test set:')
    while not qPerformance.empty():
      description, (accuracy, confusion) = qPerformance.get()
      print description, accuracy
  else:
    p = Process(name='evaluateFIN', target=evaluateQueue, args=(theta, examples['TEST'], qPerformance, 'Eventual performance on test set:'))
    pPs.append(p)
    p.start()
    for p in pPs: p.join()  # make sure all subprocesses are properly terminated

  while not qPerformance.empty():
    description, (accuracy, confusion) = qPerformance.get()
    print description, accuracy

  print 'End of training.'
  return theta


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

  #  for name in grads.keys():
  #    grads[name] = grads[name]/len(examples)+ lambdaL2*theta[name] # regularize
    else: q.put((grads, error/len(examples)))
  else:
    print 'Batch with no training examples?!'
    q.put((None,None))

