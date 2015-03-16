from __future__ import division

import os, os.path
from nltk import tree
import numpy as np
import random
import re
from NLRNN import *
import numericalGradient2 as ng
import sys

def gradientCheck(network, theta, target):
  network.forward(theta)
  grad = network.backprop(theta, target)
  numgrad = ng.numericalGradient(network,theta,target)
  if np.array_equal(numgrad,grad):
    print 'numerical and analytical gradients are equal.'
    return None
  globaldiff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
  print 'Difference numerical and analytical gradients:', globaldiff

  names = ['M1','b1','V','M2','b2','M3','b3']
  pairedGrads = zip(unwrap(numgrad),unwrap(grad))

  for i in range(len(names)):
    a,b = pairedGrads[i]
    a = np.reshape(a,-1)
    b = np.reshape(b,-1)
    diff = np.linalg.norm(a-b)/np.linalg.norm(a+b)
    print 'Difference '+names[i]+' :', diff
#     if diff > 0.0001:
#       for j in range(len(a)):
#         print a[j], b[j]
  return globaldiff

def initialize(nwords, nrel):
  # initialize all parameters randomly using a uniform distribution over [-0.1,0.1]
  M1 = np.random.rand(dwords, 2*dwords)*.02-.01  #composition weights
  b1 = np.random.rand(dwords)*.02-.01       #composition bias
  M2 = np.random.rand(dcomparison,2*dwords)*.02-.01
  b2 = np.random.rand(dcomparison)*.02-.01       #composition bias
  M3 = np.random.rand(nrel,dcomparison)
  b3 = np.random.rand(nrel)*.02-.01       #composition bias
  V = np.random.rand(nwords,dwords)*.02-.01
  theta = wrap((M1,b1,V,M2,b2,M3,b3))
  return theta

def getData(corpusdir, relations):
  print 'Reading corpus...'
  examples = []
  vocabulary = ['UNK']
  for root, _, files in os.walk(corpusdir+'/data-4'):
    for f in files:
      with open(os.path.join(root,f),'r') as f:
        for line in f:
          bits = line.split('\t')
          if len(bits) == 3:
            relation, s1, s2 = bits
            # add unknown words to vocabulary
            for word in s1.split()+s2.split():
              if word !=')' and word != '(' and word not in vocabulary:
                vocabulary.append(word)
            # add training example to set
            examples.append(([tree.Tree.fromstring('('+re.sub(r"([^()\s]+)", r"(\1)", s)+')') for s in [s1,s2]],relations.index(relation)))
  # Now that the vocabulary is established, create neural networks from the examples
  networks = []
  for (trees, target) in examples:
    nw = Network.fromTrees(trees,vocabulary)
    nw.sethyperparams(len(vocabulary), dwords, dcomparison,len(relations))
    networks.append((nw,target))
  print 'Done.'
  return networks, vocabulary

def batchtrain(alpha, lambdaL2, epochs, theta, examples):

  print 'Start batch training'
  for i in range(epochs):
    grads, error = epoch(theta, examples, lambdaL2)
    theta -= alpha/len(examples) * grads
    print '\tEpoch',i, ', average error:', error, ', theta norm:', np.linalg.norm(theta)
  print 'Done.'
  return theta

def adagrad(lambdaL2, alpha, epochs,theta, examples):
  print 'Start adagrad training'
  historical_grad = np.zeros_like(theta)

  #while not converged:
  for i in range(epochs):
    grad, error = epoch(theta, examples, lambdaL2)
    historical_grad += np.multiply(grad,grad)
    adjusted_grad = np.divide(grad,np.sqrt(historical_grad)+1e-6)
    theta = theta - alpha*adjusted_grad
    print '\tEpoch',i, ', average error:', error, ', theta norm:', np.linalg.norm(theta)
  print 'Done.'
  return theta

def SGD(lambdaL2, alpha, epochs, theta, data):
  print 'Start SGD training with minibatches'
  historical_grad = np.zeros_like(theta)
#  while not converged:
  for i in range(epochs):
    minibatch = random.sample(data, 32)
    grad, error = epoch(theta, minibatch, lambdaL2)
    historical_grad += np.multiply(grad,grad)
    adjusted_grad = np.divide(grad,np.sqrt(historical_grad)+1e-6)
    theta = theta - alpha*adjusted_grad
    print '\tIteration', i, ', average error:', error, ', theta norm:', np.linalg.norm(theta)
  return theta


def epoch(theta, examples, lambdaL2):
  grads = np.zeros_like(theta)
  error = 0
  for (network, target) in examples:
    network.forward(theta)
    grads += network.backprop(theta,target)
    error += network.error(theta, target)
  error = error/ len(examples) + lambdaL2/2 * np.linalg.norm(theta)
  grads += lambdaL2*theta
  return grads, error


def evaluate(theta, testData):
  true = 0
  for (network, target) in testData:
    prediction = network.predict(theta)
    if prediction == target:
      true +=1
  print 'Accuracy:', true/len(testData)
  return true/len(testData)

def main(args):
  if len(args)== 0:
    corpusdir = 'C:/Users/Sara/AI/thesisData/vector-entailment-ICLR14-R1'
  else:
    corpusdir = args[0]
  if not os.path.isdir(corpusdir):
    print 'no data found at', corpusdir
    sys.exit()

  # set hyperparameters
  global dwords, dcomparison
  dwords = 16
  dcomparison = 45
  alpha = 0.2
  lambdaL2 = 0.0002
  epochs = 5#50


  relations = ['<','>','=','|','^','v','#']
  examples,vocabulary = getData(corpusdir, relations)
  np.random.shuffle(examples)
  nTest = len(examples)//5
  trainData = examples[:nTest]
  testData = examples[nTest:]

  theta = initialize(len(vocabulary),len(relations))
  print 'Parameters initialized. Theta norm:', np.linalg.norm(theta)

  thetaBatch = batchtrain(alpha, lambdaL2, epochs, np.copy(theta), trainData)
  evaluate(thetaBatch,testData)

  thetaAda = adagrad(lambdaL2, alpha, epochs, np.copy(theta), trainData)
  evaluate(thetaAda,testData)

  thetaSGD = SGD(lambdaL2, alpha, epochs, np.copy(theta), trainData)
  evaluate(thetaSGD,testData)

  #network,target = examples[0]
  #gradientCheck(network,theta, target)
if __name__ == "__main__":
   main(sys.argv[1:])