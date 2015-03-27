from __future__ import division
import nltk
from nltk.tree import Tree
import numpy as np
import sys
from collections import defaultdict
global grammarBased



class Node:
  def __init__(self,children,cat, nonlinearity):
    self.cat = cat
    self.children = children
    self.nonlin = nonlinearity
  def backprop(self,delta,theta,gradients):
    childrenas = np.concatenate([child.a for child in self.children])
    childrenads = np.concatenate([child.ad for child in self.children])
    # compute delta to backprop and backprop it
    deltaB = np.split(np.multiply(np.transpose(theta[self.cat+'M']).dot(delta),childrenads),len(self.children))
    [self.children[i].backprop(deltaB[i], theta, gradients) for i in range(len(self.children))]
    # update gradients for this node
    gradients[self.cat+'M']+= np.outer(delta,childrenas)
    gradients[self.cat+'B']+=delta

  def forward(self,theta):
    inputsignal = np.concatenate([child.forward(theta) for child in self.children])
    M= theta[self.cat+'M']
    b= theta[self.cat+'B']
    self.z = M.dot(inputsignal)+b
    self.a, self.ad = activate(self.z,self.nonlin)
    return self.a

  def __str__(self):
    if self.cat == 'comparison': return '['+'] VS ['.join([str(child) for child in self.children])+']'
    elif self.cat == 'softmax': return ''.join([str(child) for child in self.children])
    else: return '('+' '.join([str(child) for child in self.children])+')'

class Leaf(Node):
  def __init__(self, cat, index, word=''):
    Node.__init__(self,[],cat,'identity')
    self.index = index
    self.word = word
  def forward(self,theta):
    self.z = theta[self.cat][self.index]
    if self.nonlin!= 'identity': print 'activation should be identity'
    self.a, self.ad = activate(self.z,self.nonlin)
    return self.a
  def backprop(self,delta, theta, gradients):
    gradients[self.cat][self.index] += delta
  def __str__(self):
    return self.word

class Top(Node):
  def backprop(self,theta,target):
    # initialize gradients
    gradients = np.zeros_like(theta)
    # determine delta and call inherited backprop function
    delta = np.array(self.a, copy = True)
    delta[target] -=1   # Phong said 1-delta[trueRel], but that did not work
    Node.backprop(self,delta,theta, gradients)
    return gradients

  def error(self,theta, target, recompute = True):
    if recompute: self.forward(theta)
    return -np.log(self.a[target])

  def predict(self, theta, recompute = True):
    if recompute: self.forward(theta)
    return self.a.argmax(axis=0)

def activate(vector, nonlinearity):
  if nonlinearity =='identity':
    act = vector
    der = np.ones_like(act)
  elif nonlinearity =='tanh':
    act = np.tanh(vector)
    der = 1- np.multiply(act,act)
  elif nonlinearity =='ReLU':
    act = np.array([max(x,0)+0.01*min(x,0) for x in vector])
    der = np.array([1*(x>=0) for x in vector])
  elif nonlinearity =='sigmoid':
    act = 1/(1+np.exp(-1*vector))
    der = act * (1 - act)
  elif nonlinearity =='softmax':
    e = np.exp(vector)
    act = e/np.sum(e)
    der = np.ones_like(act)#this is never used
  else:
    print 'no familiar nonlinearity:', nonlinearity,'. Used identity.'
    act = vector
    der = np.ones(len(act))
  return act, der

def gradientCheck(theta, network, target):
  network.forward(theta)
  grad = network.backprop(theta, target)
  numgrad = numericalGradient(theta,network,target)
  gradflat = np.array([])
  numgradflat = np.array([])
  for name in theta.dtype.names:
    ngr = np.reshape(numgrad[name],-1)
    gr = np.reshape(grad[name],-1)
    if np.array_equal(gr,ngr): diff = 0
    else: diff = np.linalg.norm(ngr-gr)/(np.linalg.norm(ngr)+np.linalg.norm(gr))
    print 'Difference '+name+' :', diff
    gradflat = np.append(gradflat,gr)
    numgradflat = np.append(numgradflat,ngr)
  print 'Difference overall:', np.linalg.norm(numgradflat-gradflat)/(np.linalg.norm(numgradflat)+np.linalg.norm(gradflat))


def numericalGradient(theta, network, target):
  epsilon = 0.0001
  numgrad = np.zeros_like(theta)
  for name in theta.dtype.names:
  # create an iterator to iterate over the array, no matter its shape
      it = np.nditer(theta[name], flags=['multi_index'])
      while not it.finished:
        i = it.multi_index
        old = theta[name][i]
        theta[name][i] = old + epsilon
        errorPlus = network.error(theta,target)
        theta[name][i] = old - epsilon
        errorMin = network.error(theta,target)
        d =(errorPlus-errorMin)/(2*epsilon)
        numgrad[name][i] = d
        theta[name][i] = old  # restore theta
        it.iternext()
  return numgrad