from __future__ import division
import numpy as np
import random

def thetaNorm(theta):
  names = theta.dtype.names
  return sum([np.linalg.norm(theta[name]) for name in names])/len(names)

def evaluate(theta, testData):
  true = 0
  for (network, target) in testData:
    prediction = network.predict(theta)
    if prediction == target:
      true +=1
#    else: print 'mistake in (', network, '), true:',target
  print 'Accuracy:', true/len(testData)
  return true/len(testData)


def batchtrain(alpha, lambdaL2, epochs, theta, examples):
  print 'Start batch training'
  for i in range(epochs):
    grads, error = epoch(theta, examples, lambdaL2)
    theta -= alpha/len(examples) * grads
    print '\tEpoch',i, ', average error:', error, ', theta norm:', thetaNorm(theta)
  print 'Done.'
  return theta

def adagrad(lambdaL2, alpha, epochs,theta, examples):
  print 'Start adagrad training'
  historical_grad = np.zeros_like(theta)

  #while not converged:
  for i in range(epochs):
    grad, error = epoch(theta, examples, lambdaL2)
    historical_grad += np.multiply(grad,grad)
    adjusted_grad = np.divide(grad,np.sqrt(historical_grad)+0.001)
    theta = theta - alpha*adjusted_grad
    print '\tEpoch',i, ', average error:', error, ', theta norm:', thetaNorm(theta)
  print 'Done.'
  return theta

def SGD(lambdaL2, alpha, epochs, theta, data):
  print 'Start SGD training with minibatches'
  historical_grad = np.zeros_like(theta)
#  while not converged:
  for i in range(epochs):
    minibatch = random.sample(data, 32)
    grad, error = epoch(theta, minibatch, lambdaL2)
    for name in historical_grad.dtype.names:
      historical_grad[name] += np.square(grad[name])
      theta[name] = theta[name] - alpha*np.divide(grad[name],np.sqrt(historical_grad[name])+1e-6)
    print '\tIteration', i, ', average error:', error, ', theta norm:', thetaNorm(theta)
  return theta

def epoch(theta, examples, lambdaL2):
  grads = np.zeros_like(theta)
  regularization = lambdaL2/2 * thetaNorm(theta)**2
  error = 0
  nans = ''
  for (network, target) in examples:
    network.forward(theta)
    e = network.error(theta, target, recompute = False)
    dgrads = network.backprop(theta,target)
    for name in grads.dtype.names:
      grads[name] += dgrads[name]

    if np.isnan(e): nans += str(target)
    else: error += e
  print 'nans:',nans
  error = error/ len(examples) + regularization
  for name in grads.dtype.names:
    grads[name] = grads[name]/len(examples)+ lambdaL2*theta[name]
  return grads, error
