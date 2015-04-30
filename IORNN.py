from __future__ import division
import nltk
from nltk.tree import Tree
import numpy as np
import sys
from collections import defaultdict
import random
#import copy

import activation

class Node:
  def __init__(self, children, cat, actI,actO):
    self.cat = cat
    self.actI = actI
    self.actO = actO
    self.children = children
    if len(children)==1:
      self.children[0].setRelatives(self,None)
    if len(children)==2:
      self.children[0].setRelatives(self,self.children[1])
      self.children[1].setRelatives(self,self.children[0])
    if len(children)>2:
      print 'Something is rotten'
    self.parent = None
    self.sibling = None
  def recomputeNW(self, theta):
    try: self.parent.recomputeNW(theta)
    except:
      self.inner(theta)
      self.outer(theta)

  def activateNW(self,theta):
    self.inner(theta)
    self.outer(theta)

  def setRelatives(self, parent, sibling):
#    print 'self: [', self,'] (', self.cat,  '). Parent: [',parent, ']. Sibling: [', sibling,']'
    self.parent = parent
    self.sibling = sibling

  def backpropInner(self,delta,theta,gradients):
#    print 'backpropInner', self
    childrenas = np.concatenate([child.innerA for child in self.children])
    childrenads = np.concatenate([child.innerAd for child in self.children])
    # compute delta to backprop and backprop it
    deltaB = np.split(np.multiply(np.transpose(theta[self.cat+'IM']).dot(delta),childrenads),len(self.children))
    [self.children[i].backpropInner(deltaB[i], theta, gradients) for i in range(len(self.children))]
    # update gradients for this node
    gradients[self.cat+'IM']+= np.outer(delta,childrenas)
    gradients[self.cat+'IB']+=delta

  def backpropOuter(self, delta, theta, gradients):
#    print 'backpropOuter', self#, self.sibling
    if self.parent:
      As = self.parent.outerA
      Ads = self.parent.outerAd
      if self.sibling:
        As = np.append(As,self.sibling.innerA)
        Ads = np.append(Ads,self.sibling.innerAd)
      cat = self.parent.cat
      M = theta[cat+'OM']
#      print cat, np.shape(M), np.shape(delta)
      deltaB = np.multiply(np.transpose(M).dot(delta), Ads)
      if self.sibling:
#        print 'backproping to parent and sibling'
        deltaB = np.split(deltaB,2)
        self.parent.backpropOuter(deltaB[0], theta, gradients)
        self.sibling.backpropInner(deltaB[1], theta, gradients)
      else: self.parent.backpropOuter(deltaB, theta, gradients)
      gradients[cat+'OM']+= np.outer(delta,As)
      gradients[cat+'OB']+= delta

  def inner(self, theta):
    inputsignal = np.concatenate([child.inner(theta) for child in self.children])
    M= theta[self.cat+'IM']
    b= theta[self.cat+'IB']
    self.innerZ = M.dot(inputsignal)+b
    self.innerA, self.innerAd = activation.activate(self.innerZ,self.actI)
    return self.innerA

  def outer(self, theta):
#    print 'outer called for:', self,  'of cat', self.cat
    if not self.parent:
      self.outerZ = np.zeros_like(theta[self.cat+'IB'])
    else:
      if self.sibling: inputsignal = np.concatenate([self.parent.outerA,self.sibling.innerA])
      else: inputsignal = self.parent.outerA
      cat = self.parent.cat
      M= theta[cat+'OM']
      b= theta[cat+'OB']
      self.outerZ = M.dot(inputsignal)+b
    self.outerA, self.outerAd = activation.activate(self.outerZ,self.actO)
    [child.outer(theta) for child in self.children]
#    return self.outerA

  def train(self, theta, gradients = None, target = None):
    if gradients == None: gradients = np.zeros_like(theta)
    [child.train(theta, gradients) for child in self.children]
    return gradients

  def __str__(self):
    if self.cat == 'comparison': return '['+'] VS ['.join([str(child) for child in self.children])+']'
    elif self.cat == 'softmax': return ''.join([str(child) for child in self.children])
    elif self.cat == 'u': return '<'+' '.join([str(child) for child in self.children])+'>'
    elif self.cat == 'score': return 's'
    else: return '('+' '.join([str(child) for child in self.children])+')'

class Leaf(Node):
  def __init__(self, cat, index, actO, word=''):
    children = [Node([Node([], 'score', 'identity','identity')], 'u', 'tanh','tanh')]
    Node.__init__(self,children,cat,'identity',actO)
    children[0].setRelatives(self,self)
    # this node should behave as a sibling (produce an inner representation) and a parent (outer) to the u node
    self.index = index
    self.word = word

  def train(self, theta, gradients = None, target = None):
    if gradients == None: gradients = np.zeros_like(theta)
    nwords = len(theta['wordIM'])
    scorew = self.score(theta, False)
    for n in range(1):#3? sample size for candidates
      x = random.randint(0,nwords-1)
      while x == self.index:  x = random.randint(0,nwords-1)
      # if the candidate scores too high: backpropagate error
      scorex = self.score(theta, x, False)
      c = 1 - scorew+scorex
      if c>1:
        delta = np.array([1])
        self.children[0].children[0].backpropOuter(delta, theta, gradients)
    return gradients

  def score(self, theta, wordIndex=-1, recompute = True):
    self.recomputeNW(theta)
    # pretend the index is the candidate
    if wordIndex > 0:
      trueIndex = self.index
      self.index = wordIndex
      self.inner(theta)
      self.outer(theta)
    score = self.children[0].children[0].outerA
    # reset everything
    if wordIndex > 0:
      self.index = trueIndex
      self.inner(theta)
      self.outer(theta)
    return score[0]

  def numericalGradient(theta, nw, target = None):
  #  print 'numgrad', theta.dtype.names
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
          errorPlus = max(0,1-score0+nw.score(theta))
          theta[name][i] = old - epsilon
          errorMin = max(0,1-score0+nw.score(theta))
          d =(errorPlus-errorMin)/(2*epsilon)
          numgrad[name][i] = d
          theta[name][i] = old  # restore theta
          it.iternext()
    return numgrad

  def inner(self, theta):

    self.innerZ = theta[self.cat+'IM'][self.index]
    self.innerA, self.innerAd = activation.activate(self.innerZ,self.actI)
    return self.innerA


  def backpropInner(self,delta, theta, gradients):
#    print 'backpropInner', self
    gradients[self.cat+'IM'][self.index] += delta

  def __str__(self):
    return self.word





