from __future__ import division
import nltk
from nltk.tree import Tree
import numpy as np
import sys
from collections import defaultdict
import random

class Node:
  def __init__(self, children, cat, nonlinearity):
    self.cat = cat
    self.nonlin = nonlinearity
    self.children = children
    self.parent = None
    self.singling = None

  def setRelatives(self, parent, sibling):
    self.parent = parent
    self.sibling = sibling

  def forwardOuter(self,theta):
    [child.forwardOuter(theta) for child in self.children]

  def backpropInner(self,delta,theta,gradients):
    childrenas = np.concatenate([child.innerA for child in self.children])
    childrenads = np.concatenate([child.innerAd for child in self.children])
    # compute delta to backprop and backprop it
    deltaB = np.split(np.multiply(np.transpose(theta[self.cat+'MI']).dot(delta),childrenads),len(self.children))
    [self.children[i].backpropInner(deltaB[i], theta, gradients) for i in range(len(self.children))]
    # update gradients for this node
    gradients[self.cat+'MI']+= np.outer(delta,childrenas)
    gradients[self.cat+'BI']+=delta

  def backpropOuter(self, delta, theta, gradients):
    if self.parent:
      As = np.concatenate([self.parent.outerA,self.sibling.innerA])
      Ads = np.concatenate([self.parent.outerAd,self.sibling.innerAd])
      cat = self.parent.cat
      M = theta[cat+'MO']
      deltaB = np.split(np.multiply(np.transpose(M).dot(delta), Ads),2)
      self.parent.backpropOuter(deltaB[0], theta, gradients)
      self.sibling.backpropInner(deltaB[1], theta, gradients)
      gradients[cat+'MO']+= np.outer(delta,As)
      gradients[cat+'BO']+= delta

  def inner(self, theta):
    inputsignal = np.concatenate([child.inner(theta) for child in self.children])
    M= theta[self.cat+'MI']
    b= theta[self.cat+'BI']
    self.innerZ = M.dot(inputsignal)+b
    self.innerA, self.innerAd = activate(self.innerZ,self.nonlin)
    return self.innerA

  def outer(self, theta):
    if self.parent:
      inputsignal = np.concatenate([self.parent.outer(theta),self.sibling.inner(theta)])
      cat = self.parent.cat
      M= theta[cat+'MO']
      b= theta[cat+'BO']
      self.outerZ = M.dot(inputsignal)+b
    else:
      self.outerZ = np.zeros(25)
    self.outerA, self.outerAd = activate(self.outerZ,self.nonlin)
    return self.outerA

  def __str__(self):
    if self.cat == 'comparison': return '['+'] VS ['.join([str(child) for child in self.children])+']'
    elif self.cat == 'softmax': return ''.join([str(child) for child in self.children])
    else: return '('+' '.join([str(child) for child in self.children])+')'

class Leaf(Node):
  def __init__(self, cat, index, word=''):
    Node.__init__(self,[],cat,'identity')
    self.index = index
    self.word = word

  def forwardOuter(self,theta):
    self.outer(theta)

  def backpropOuter(self, theta, gradients):
    delta =
    Node.backpropOuter(self, delta, theta, gradients)


      cat = self.parent.cat
      M = theta[cat+'MO']
      deltaB = np.split(np.multiply(np.transpose(M).dot(delta), Ads),2)
      self.parent.backpropOuter(deltaB[0], theta, gradients)
      self.sibling.backpropInner(deltaB[1], theta, gradients)
      gradients[cat+'MO']+= np.outer(delta,As)
      gradients[cat+'BO']+= delta

  def inner(self, theta):
    self.innerZ = theta[self.cat][self.index]
    self.innerA, self.innerAd = activate(self.innerZ,self.nonlin)
    return self.innerA


  def backpropInner(self,delta, theta, gradients):
    gradients[self.cat][self.index] += delta

  def __str__(self):
    return self.word

  def score(self, theta, wordIndex=-1):
    if wordIndex <0: wordIndex = self.index
    M = theta['uM']
    b = theta['uB']
    some = M.dot(np.concatenate([self.outerA,theta['word'][wordIndex]]))+b
#    some = M[0].dot(self.outerA) + M[1].dot(theta['word'][wordIndex]) + b
    u ,ud = activate(some, self.nonlin)
    score = activate(np.transpose(theta['scoreM']).dot(u),'tanh')
    return score

# class Top(Node):
#   def __init__(self, children, cat, nonlinearity):
#     Node.__init__(self, children, cat, nonlinearity)
#     self.parent = None
#     self.sibling = None

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

def numericalGradient(theta, nw, w):
  epsilon = 0.0001
  numgrad = np.zeros_like(theta)
  score0 = nw.score(theta)
  for name in theta.dtype.names:
  # create an iterator to iterate over the array, no matter its shape
      it = np.nditer(theta[name], flags=['multi_index'])
      while not it.finished:
        i = it.multi_index
        old = theta[name][i]
        theta[name][i] = old + epsilon
        errorPlus = max(0,1-score0+nw.score(theta, w))
        theta[name][i] = old - epsilon
        errorMin = max(0,1-score0+nw.score(theta, w))
        d =(errorPlus-errorMin)/(2*epsilon)
        numgrad[name][i] = d
        theta[name][i] = old  # restore theta
        it.iternext()
  return numgrad

def compareGrad(numgrad,grad):
  gradflat=np.array([])
  numgradflat=np.array([])
  for name in numgrad.dtype.names:
    ngr = np.reshape(numgrad[name],-1)
    gr = np.reshape(grad[name],-1)
    if np.array_equal(gr,ngr): diff = 0
    else: diff = np.linalg.norm(ngr-gr)/(np.linalg.norm(ngr)+np.linalg.norm(gr))
    print 'Difference '+name+' :', diff
    gradflat = np.append(gradflat,gr)
    numgradflat = np.append(numgradflat,ngr)
  print 'Difference overall:', np.linalg.norm(numgradflat-gradflat)/(np.linalg.norm(numgradflat)+np.linalg.norm(gradflat))

def trainPredict(nw, theta,vocabulary,gradients=None):
  if gradients==None: gradients = np.zeros_like(theta)
  if isinstance(nw,Leaf):
    score0 = nw.score(theta)
    for n in range(5):
      # create candidate index unlike the observed one
      i = random.randint(0,len(vocabulary)-1)
      while i == nw.index:
        i = random.randint(0,len(vocabulary)-1)
      # if the candidate scores too high: backpropagate error
      score1 = nw.score(theta, i)
      if 1-score0+score1>0:
        delta = np.reshape(np.split(theta['uM'],2)[0].dot(theta['scoreM']),-1)

        numGrad = numericalGradient(theta, nw, i)
        nw.backpropOuter(delta, theta, gradients)
        compareGrad(numGrad,gradients)
  else: [trainPredict(child, theta, vocabulary, gradients) for child in nw.children]