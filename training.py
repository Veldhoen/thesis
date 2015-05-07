from __future__ import division
import numpy as np
import random
import IORNN
from collections import defaultdict, Counter
import pickle, os

def thetaNorm(theta):
  names = theta.dtype.names
  return sum([np.linalg.norm(theta[name]) for name in names])/len(names)

def evaluate(theta, testData, amount=1):
  if isinstance(testData[0], IORNN.Node):
    return evaluateIOUS(theta,testData, amount),None
  true = 0
  confusion = defaultdict(Counter)
  for nw, tar in testData:
    pred = nw.predict(theta)
    confusion[tar][pred] += 1
    if pred == tar: true +=1
  return true/len(testData), confusion

def evaluateIOUS(theta,testData, amount=1):
  if amount<= 1: n = int(amount*len(testData))
  else: n = amount
  ranks = 0
  num = 0
  nwords = len(theta['wordIM'])
  for nw in random.sample(testData,n):
    nw.activateNW(theta)
    leaves = nw.leaves()
    # we don't expect the network to make true predictions
    # for such small trees
    if len(leaves)<3: continue

    for leaf in random.sample(leaves,2):
      scores = [leaf.score(theta,x,False)[0] for x in xrange(nwords)]
      ranking = np.array(scores).argsort().argsort()+1
      ranks+= ranking[leaf.index]
      num +=1
  return ranks/(nwords*num)

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


def batchtrain(alpha, lambdaL2, epochs, theta, examples):
  print 'Start batch training'
  for i in xrange(epochs):
    grads, error = epoch(theta, examples, lambdaL2)
    for name in theta.dtype.names:
      theta[name] -= alpha/len(examples) * grads[name]
    print '\tEpoch',i, ', average error:', error, ', theta norm:', thetaNorm(theta)
  print 'Done.'
  return theta

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
def SGD(lambdaL2, alpha, epochs, theta, data, testData, relations, batchsize =0):
  print 'Start SGD training with random minibatches'
  historical_grad = np.zeros_like(theta)
  accuracy, confusion = evaluate(theta,testData)
  if batchsize == 0: batchsize = len(data)
#  while not converged:
  for i in xrange(epochs):
    print 'Iteration', i ,', Performance on test sample:', accuracy
    for batch in xrange(len(data)//batchsize):
      minibatch = random.sample(data, batchsize)

      grad, error = epoch(theta, minibatch, lambdaL2)
      updateTheta(theta, grad,historical_grad,alpha)
      if batch % 10 == 0:
        print '\tBatch', batch, ', average error:', error, ', theta norm:', thetaNorm(theta)
    accuracy, confusion = evaluate(theta,testData,0.1)
  accuracy, confusion = evaluate(theta,testData)
  print 'Training terminated. Performance on entire test set:', accuracy
  print confusionString(confusion, relations)
  with open(os.path.join('models','flickrIO.pik'), 'wb') as f:
    pickle.dump(theta, f, -1)

# the minibatches are a random but true partition of the data
def bowmanSGD(lambdaL2, alpha, epochs, theta, data, testData, relations, batchsize =0):
  print 'Start SGD training with true minibatches'
  if batchsize == 0: batchsize = len(data)
  historical_grad = np.zeros_like(theta)
#  while not converged:
  accuracy, confusion = evaluate(theta,testData)
  for i in xrange(epochs):
    print 'Iteration', i ,', Performance on test sample:', accuracy
    print confusionString(confusion, relations)

    random.shuffle(data) # randomly split the data into parts of batchsize
    for batch in xrange(len(data)//batchsize):
      minibatch = data[batch*batchsize:(batch+1)*batchsize]
      grad, error = epoch(theta, minibatch, lambdaL2)
      updateTheta(theta, grad,historical_grad,alpha)
      if batch % 10 == 0:
        print '\tBatch', batch, ', average error:', error, ', theta norm:', thetaNorm(theta)
    accuracy, confusion = evaluate(theta,testData, 0.1)
  accuracy, confusion = evaluate(theta,testData, 1)
  print 'Training terminated. Performance on entire test set:', accuracy
  print confusionString(confusion, relations)
  with open(os.path.join('models','flickrIO.pik'), 'wb') as f:
    pickle.dump(theta, f, -1)





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
  for nw in examples:
    dgrads,derror = nw.train(theta)
    error += derror
    for name in grads.dtype.names:
      grads[name] += dgrads[name]

  for name in grads.dtype.names:
    grads[name] = grads[name]/len(examples)+ lambdaL2*theta[name] # regularize
  return grads, error/len(examples)