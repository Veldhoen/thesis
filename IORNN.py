from __future__ import division
#import nltk
#from nltk.tree import Tree
import numpy as np
import sys
from collections import defaultdict
import random
#import copy
#from theano import sparse
import activation
from scipy import sparse

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
#      self.children[0].setRelatives(None,self.children[1])
#      self.children[1].setRelatives(None,self.children[0])

    if len(children)>2:
      print 'Something is rotten'
    self.setRelatives(None,None)

  ''' activate the entire network'''
  def recomputeNW(self, theta):
#    print 'recompute'
    try: self.parent.recomputeNW(theta)
    except:
      self.activateNW(theta)

  ''' activate this node and everything below it'''
  def activateNW(self,theta):
#    if self.cat != 'word': print 'activate', self.cat
    self.inner(theta)
    self.outer(theta)

  '''return a list of all the leaf nodes under this node'''
  def leaves(self):
    return sum([child.leaves() for child in self.children],[])



  def setRelatives(self, parent, sibling):
#    print 'self: [', self,'] (', self.cat,  '). Parent: [',parent, ']. Sibling: [', sibling,']'
    self.parent = parent
    self.sibling = sibling

  def backpropInner(self,delta,theta,gradients):
#    print 'node.backpropInner', self, np.shape(delta), delta
    childrenas = np.concatenate([child.innerA for child in self.children])
    childrenads = np.concatenate([child.innerAd for child in self.children])
    # compute delta to backprop and backprop it
    deltaB = np.split(np.multiply(np.transpose(theta[self.cat+'IM']).dot(delta),childrenads),len(self.children))
    [self.children[i].backpropInner(deltaB[i], theta, gradients) for i in xrange(len(self.children))]
    # update gradients for this node
    gradients[self.cat+'IM']+= np.outer(delta,childrenas)
    gradients[self.cat+'IB']+=delta

  def backpropOuter(self, delta, theta, gradients):
#    print 'backpropOuter', self.cat, self#, self.sibling
    # input: OUTER of the parent
    #        INNER of the sibling (if it exists)

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
#        print '\t backproping to parent and sibling'
        deltaB = np.split(deltaB,2)
        self.parent.backpropOuter(deltaB[0], theta, gradients)
        self.sibling.backpropInner(deltaB[1], theta, gradients)
      else:
#        print '\t backproping to parent only'
        self.parent.backpropOuter(deltaB, theta, gradients)
      gradients[cat+'OM']+= np.outer(delta,As)
      gradients[cat+'OB']+= delta
#    else: print 'backpropouter stops: top node'


  def inner(self, theta):
#    print 'node.inner', self.cat
#    inputsignal = np.concatenate([child.inner(theta) for child in self.children])
    inputs = [child.inner(theta) for child in self.children]

    inputsignal = np.concatenate(inputs)
    M= theta[self.cat+'IM']
    b= theta[self.cat+'IB']
    try: self.innerZ = M.dot(inputsignal)+b
    except:
      print self.cat, ', matrix:',M.shape, ', input:' ,inputsignal.shape
      for c in self.children: print c.cat, c.innerA.shape
      sys.exit()
    self.innerA, self.innerAd = activation.activate(self.innerZ,self.actI)
    return self.innerA

  def outer(self, theta):
#    print 'node.outer', self.cat
#    print 'outer called for:', self,  'of cat', self.cat
    if not self.parent:
#      size = np.size(theta[self.cat+'OB'])
#      self.outerZ = np.random.rand(size)*0.2-0.1

      self.outerZ = np.zeros_like(theta[self.cat+'OB'])
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
#    if gradients is None: gradients = np.zeros_like(theta)
    print 'network',self,'start training with target', target
    if gradients is None:
      gradients = theta.zeros_like()
    self.activateNW(theta)
    error = np.mean([leaf.trainWords(theta, gradients,target) for leaf in self.leaves()])
    if error>2: print 'nw error:', error

    return gradients,error

  def error(self,theta, x=0, recompute = False):
#    print 'nw.error called for', x
#    if not type(x) is int: print 'nw.score called for', x
#    return sum([leaf.score(theta,x,recompute) for leaf in self.leaves()])
     summedC = 0
     for leaf in self.leaves():
       scorew = leaf.score(theta, -1, recompute)[0]
       scorex = leaf.score(theta, x, recompute)[0]
       c= max(0,1 - scorew+scorex)
#  #     print 'c:',c
       summedC += c
     return summedC

  def predict(self, theta):
    return max([c.predict(theta) for c in self.children])

#  def score(self, theta,

  def __str__(self):
    if self.cat == 'comparison': return '['+'] VS ['.join([str(child) for child in self.children])+']'
    elif self.cat == 'softmax': return ''.join([str(child) for child in self.children])
    elif self.cat == 'u': return '<'+' '.join([str(child) for child in self.children])+'>'
    elif self.cat == 'score': return 's'
    else: return '('+' '.join([str(child) for child in self.children])+')'

  def numericalGradient(self, theta, target = None):
    nw = self
#    nw.activateNW(theta)
    print 'Computing numerical gradient for target', target
  #  print 'numgrad', theta.dtype.names
    epsilon = 0.0001
    numgrad = theta.zeros_like(False)
#    score0 = nw.score(theta, target, True)
    error0 = nw.error(theta,target,True)
#    print 'score0:', score0
#    print 'error:', error0
    for name in theta.keys():
        print '\t',name
    # create an iterator to iterate over the array, no matter its shape
        it = np.nditer(theta[name], flags=['multi_index'])
        while not it.finished:
          nw.recomputeNW(theta)
          i = it.multi_index
          old = theta[name][i]
          theta[name][i] = old + epsilon
 #         errorPlus = max(0,1-score0+nw.score(theta,target))
          errorPlus=nw.error(theta,target,True)
          theta[name][i] = old - epsilon
#          errorMin = max(0,1-score0+nw.score(theta,target))
          errorMin=nw.error(theta,target,True)
          d =(errorPlus-errorMin)/(2*epsilon)
#          if d!=0: print '\t\tchange gradient, diff:', errorPlus-errorMin
          numgrad[name][i] = d
          theta[name][i] = old  # restore theta
          it.iternext()
    return numgrad


class Leaf(Node):
  def __init__(self, cat, index, actO, word=''):
#    children = [Node([Node([], 'score', 'identity','identity')], 'u', 'tanh','tanh')]
    children = [Node([Node([], 'score', 'sigmoid','sigmoid')], 'u', 'tanh','tanh')]
#    children = [Node([Node([], 'score', 'tanh','tanh')], 'u', 'tanh','tanh')]
    Node.__init__(self,children,cat,'identity',actO)
    children[0].setRelatives(self,self)
    children[0].children[0].setRelatives(children[0],None)
    # this node should behave as a sibling (produce an inner representation) and a parent (outer) to the u node
    self.index = index
    self.word = word

  def trainWords(self, theta, gradients, target = None):
    nwords = len(theta[self.cat+'IM'])

    # pick a candidate x different from own index
    if target is None:
      x = self.index
      while x == self.index:  x = random.randint(0,nwords-1)
    else: x = target

    # compute error for chosen candidate
    error = 1 - self.score(theta, recompute= False)+self.score(theta, x, False)

    if True: #c>1: # if the candidate scores too high: backpropagate error
      # backpropagate through observed node
      delta = -1*self.children[0].children[0].outerAd[0]
      self.children[0].children[0].backpropOuter(delta, theta, gradients)

      # backpropagate through candidate
      original = self.index
      self.index = x
      self.activateNW(theta)
      delta = self.children[0].children[0].outerAd[0]
      self.children[0].children[0].backpropOuter(delta, theta, gradients)

      # restore observed node
      self.index = original
      self.activateNW(theta)

    if error>2: print 'leaf error:', c, self.word
    return error

  def leaves(self):
    return [self]

  def score(self, theta, wordIndex=-1, recompute = True):
    if recompute: self.recomputeNW(theta)
    # pretend the index is the candidate
    if wordIndex > -1:
      trueIndex = self.index
      self.index = wordIndex
      self.activateNW(theta)
    # compute score
    score = self.children[0].children[0].outerA

    # reset everything
    if wordIndex > -1:
      self.index = trueIndex
      self.activateNW(theta)
    return score[0]

  def predict(self, theta):
    if self.cat == 'rel':
      scores = []
      for index in xrange(len(theta['relIM'])):
        scores.append(self.score(theta,index, False))
      return scores.index(max(scores))
    else: return None


  def inner(self, theta):
#    print 'leaf.inner', self.cat
    if self.cat == 'rel': print 'Leaf.inner', self.cat, self.index
    self.innerZ = np.asarray(theta[self.cat+'IM'][self.index]).flatten()
    # after theta is updated, the wordIM has become a matrix instead of a 2D-array.
    # therefore, the innerZ is of dimension (1,5) rather than (5,) as it used to be.
    # hence the flattening
    self.innerA, self.innerAd = activation.activate(self.innerZ,self.actI)
#    print self.cat, self.innerZ.shape, self.innerA.shape
    return self.innerA

  def backpropInner(self,delta, theta, gradients):
#    print 'leaf.backpropInner', self, np.shape(delta), delta
    d = len(delta)
    row = np.array([self.index]*d)
    col = np.arange(d)
#    print delta.shape, d
    deltaM = sparse.csc_matrix((delta,(row,col)),shape=np.shape(theta[self.cat+'IM']))
#    deltaM = np.zeros_like(theta['wordIM'])
    deltaM[self.index] = delta
    gradients[self.cat+'IM'] = gradients[self.cat+'IM']+(deltaM)

  def __str__(self):
    return self.word





