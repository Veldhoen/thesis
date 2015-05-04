from __future__ import division
import numpy as np
import random
import IORNN
from collections import defaultdict, Counter

def thetaNorm(theta):
  names = theta.dtype.names
  return sum([np.linalg.norm(theta[name]) for name in names])/len(names)

# def evaluate(theta, testData):
#   true = 0
#   confusion = defaultdict(Counter)
#   for (network, target) in testData:
#     prediction = network.predict(theta)
#     confusion[target][prediction] += 1
#     if prediction == target:
#       true +=1
# #    else: print 'mistake in (', network, '), true:',target
#   return true/len(testData), confusion

def evaluate(theta, testData):
  if isinstance(testData, list):
    return evaluateIOUS(theta,testData)
  
  true = 0
  confusion = defaultdict(Counter)

  for nw, tar in testData:
    pred = nw.predict(theta)

#    print tar, pred, pred == tar
    confusion[tar][pred] += 1
    if pred == tar: true +=1
  return true/len(testData), confusion

def evaluateIOUS(theta,testData):
  ranks = 0
  for nw in testData:
    for leaf in nw.leaves():
      scores = np.zeros(nwords)
      for x in range(nwords):
        scores[x] = nw.score(theta,x)
        ranks+= nwords-scores.argsort().argsort()[leaf.index]
  return ranks/(len(testData)*nwords)

def confusionString(confusion, relations):
  st = '\t'+'\t'.join(relations)
  for tar in range(len(relations)):
    st+='\n'+relations[tar]
    for pred in range(len(relations)):
      st+='\t'+str(confusion[tar][pred])
    tot = sum(confusion[tar].values())
    st+= '\t'+str(tot)
    if tot>0: st+= '\t'+str(round(confusion[tar][tar]/tot,3))
    else: st+= '\t0'
  st+='\n'
  next = '\n'
  for pred in range(len(relations)):
    tot = sum([confusion[tar][pred] for tar in range(len(relations))])
    st += '\t'+str(tot)
    if tot > 0: next+= '\t'+str(round(confusion[pred][pred]/tot,3))
    else: next+= '\t0'
  st+= next
  return st


def batchtrain(alpha, lambdaL2, epochs, theta, examples):
  print 'Start batch training'
  for i in range(epochs):
    grads, error = epoch(theta, examples, lambdaL2)
    for name in theta.dtype.names:
      theta[name] -= alpha/len(examples) * grads[name]
    print '\tEpoch',i, ', average error:', error, ', theta norm:', thetaNorm(theta)
  print 'Done.'
  return theta

def SGD(lambdaL2, alpha, epochs, theta, data, batchsize = 0):
#  print 'Start SGD training with minibatches'
  historical_grad = np.zeros_like(theta)
#  while not converged:
  for i in range(epochs):
    if batchsize > 0: minibatch = random.sample(data, batchsize)
    else: minibatch = data
    grad, error = epoch(theta, minibatch, lambdaL2)
    for name in historical_grad.dtype.names:
      historical_grad[name] += np.square(grad[name])
      theta[name] = theta[name] - alpha*np.divide(grad[name],np.sqrt(historical_grad[name])+1e-6)
    print '\tIteration', i, ', average error:', error, ', theta norm:', thetaNorm(theta)

def bowmanSGD(lambdaL2, alpha, epochs, theta, data, testData, relations, batchsize =0):
#  print 'Start SGD training with minibatches'
  if batchsize == 0: batchsize = len(data)
  historical_grad = np.zeros_like(theta)
#  while not converged:
  accuracy, confusion = evaluate(theta,testData)
  #print 'Accuracy (before training):', accuracy
  #print confusionString(confusion, relations)
  for i in range(epochs):
    print 'Iteration', i ,', Accuracy:', accuracy
    print confusionString(confusion, relations)
    # randomly split the data into parts of batchsize
    random.shuffle(data)
    for batch in range(len(data)//batchsize):
      minibatch = data[batch*batchsize:(batch+1)*batchsize]
      grad, error = epoch(theta, minibatch, lambdaL2)
      for name in historical_grad.dtype.names:
        historical_grad[name] += np.square(grad[name])
        theta[name] = theta[name] - alpha*np.divide(grad[name],np.sqrt(historical_grad[name])+1e-6)
      if batch % 10 == 0:
        print '\tBatch', batch, ', average error:', error, ', theta norm:', thetaNorm(theta)
    accuracy, confusion = evaluate(theta,testData)
#  accuracy, confusion = evaluate(theta,testData)

  print 'Training terminated. Accuracy:', accuracy
  print confusionString(confusion, relations)





# def epoch(theta, examples, lambdaL2):
#   grads = np.zeros_like(theta)
#   regularization = lambdaL2/2 * thetaNorm(theta)**2
#   error = 0
#   for (network, target) in examples:
#     network.forward(theta)
#     error += network.error(theta, target, recompute = False)
#     dgrads = network.backprop(theta,target)
#     for name in grads.dtype.names:
#       grads[name] += dgrads[name]
#   error = error/ len(examples) + regularization
#   for name in grads.dtype.names:
#     grads[name] = grads[name]/len(examples)+ lambdaL2*theta[name]
#   return grads, error

def epoch(theta, examples, lambdaL2):
  grads = np.zeros_like(theta)
  regularization = lambdaL2/2 * thetaNorm(theta)**2
  error = 0
  for nw, target in examples:
#     try: 
#        = ex
#       error += nw.error(theta, target, recompute = False)
#       print 'this is a regular RNN'
#     except:
#       nw = ex
#       target = None
#       print 'this is a IORNN'
    dgrads = nw.train(theta,target)
    for name in grads.dtype.names:
      grads[name] += dgrads[name]

  for name in grads.dtype.names:
    grads[name] = grads[name]/len(examples)+ lambdaL2*theta[name] # regularize
  return grads, error