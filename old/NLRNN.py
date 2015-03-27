from __future__ import division

from nltk import tree
import numpy as np
import sys

# NB according to numerical gradient checking the backprop is mostly OK,
# but sometimes the gradient of b2 has errors - there is no clear pattern
# in the nature of these errors. The rest of the gradients are just fine (difference e-09 or smaller)..

'''
Network is the main class: the 'outside world' needs only interfere with this.
A Network has a pointer to a Comparisonlayer, which has pointers to two RNNs.
Functions for the outside world:
- forward(theta):          activates the entire network with the given parameters
- backprop(theta, target): computes and backpropagates the error,
                           returns the gradients in the same shape as theta
- error (theta, target):   returns the error of the network with given parameters
- predict(theta):          returns the predicted class (integer) with given parameters
- str():                   returns a string representation of the network + its prediction (if activated)
'''

class Network:
  def __init__(self, comparisonlayer):
    self.comparison = comparisonlayer #
  def sethyperparams(self, nw, dw, di, dc, nr):
    global nwords, dwords, dint, dcomparison,numrel
    nwords = nw
    dwords = dw
    dint  = di
    dcomparison = dc
    numrel = nr

  @classmethod
  def fromTrees(self,trees,words):
    return Network(Comparisonlayer([RNN(tree[0], words) for tree in trees]))

  def predict(self,theta):
    self.forward(theta)
    return self.a.argmax(axis=0)

  def forward(self, theta):
    M1,b1,V,Mw,M2,b2, M3, b3 = unwrap(theta)
    self.z = M3.dot(self.comparison.forward(M1,b1,V,Mw,M2,b2)) + b3
    self.a = np.exp(self.z) /    sum(np.exp(self.z))
    return self.a
  def backprop(self,theta,trueRelation):
    M1,b1,V,Mw,M2,b2, M3, b3 = unwrap(theta)
    # compute this node's delta
    delta = self.a
    delta[trueRelation] = -1+delta[trueRelation]   # Phong said 1-delta[trueRel], but that did not work
    # compute gradients for M3 and b3
    gradM3 = np.outer(delta,self.comparison.a)
    gradb3 = delta

    # compute delta to backpropagate
    deltaB = np.multiply(np.transpose(M3).dot(delta),self.comparison.ad)
    # backpropagate to retrieve other gradients
    gradM1, gradb1, gradV, gradMw, gradM2, gradb2 = self.comparison.backprop(deltaB, M1,b1,V,Mw,M2,b2)
    return wrap((gradM1, gradb1, gradV, gradMw, gradM2, gradb2, gradM3, gradb3))
  def error(self, theta, trueRelation):
    self.forward(theta)
    return -np.log(self.a[trueRelation])

  def __str__(self):
    rels = ['<','>','=','|','^','v','#']
    st = str(self.comparison)
    try: st += ', pred: '+ rels[self.a.argmax(axis=0)]
    except : True
    return st

class RNN:
  def __init__(self,tree,words):
    self.label = tree.label
    if len(tree) == 1:# and tree.height > 2:
      self.word = tree[0]
      try: self.index = words.index(self.word.lower())
      except: self.index = 0
      self.children = []
#      print tree
#      sys.exit()
#      self.children = [RNN(child,words) for child in tree[0]]
    else: self.children = [RNN(child,words) for child in tree]

  def forward(self, M,b,V,Mw):
    if len(self.children) > 0:
      self.z = M.dot(np.concatenate([child.forward(M,b,V,Mw) for child in self.children]))+b
      self.a, self.ad = tanh(self.z)
    else:
      self.z = V[self.index]
      self.a, self.ad = identity(self.z)
    return self.a
  def backprop(self, delta, M, b, V, Mw):
    if len(self.children) > 1:
      childrenas = np.concatenate([rnn.a for rnn in self.children])
      gradM = np.outer(delta,childrenas)
      gradb = delta

      childrenads = np.concatenate([rnn.ad for rnn in self.children])
      # compute delta to backpropagate
      deltaB = np.split(np.multiply(np.transpose(M).dot(delta),childrenads),2)
      # backpropagate to retrieve other gradients
      grad0M, grad0b, grad0V,grad0Mw= self.children[0].backprop(deltaB[0], M,b,V,Mw)
      grad1M, grad1b, grad1V,grad1Mw= self.children[1].backprop(deltaB[1], M,b,V,Mw)
      gradM += grad0M + grad1M
      gradb += grad0b + grad1b
      gradV  = grad0V  + grad1V
      gradMw = grad0Mw + grad1Mw
    else:
      gradM = np.zeros_like(M)
      gradb = np.zeros_like(b)
      if len(self.children) > 0:
        if dwords == dint: gradMw = np.zeros_like(Mw)
        else:
          gradMw = np.outer(delta,self.children[0].a)
          gradBw = delta         #NB add gradBw everywhere! 
        deltaB = np.multiply(np.transpose(Mw).dot(delta),self.children[0].ad)
        gradV = self.children[0].backprop(deltaB, M,b,V,Mw)
      else:
        gradMw = np.zeros_like(Mw)
        gradV = np.zeros_like(V)
        gradV[self.index] = delta   # this is where the shape is corrupted

    return gradM, gradb, gradV, gradMw
  def __str__(self):
    if len(self.children) > 0:
      return '['+','.join([str(child) for child in self.children])+']'
    else:
      if self.word: return self.word
      else:         return str(self.wordIndex)


class Comparisonlayer:
  def __init__(self,rnns):
    self.rnns = rnns
  def forward(self, M1,b1,V,Mw,M2,b2):
    self.z = M2.dot(np.concatenate([rnn.forward(M1,b1,V,Mw) for rnn in self.rnns]))+b2
    self.a, self.ad = ReLU(self.z)
    return self.a

  def backprop(self,delta,M1,b1,V,Mw,M2,b2):
    childrenas = np.concatenate([rnn.a for rnn in self.rnns])
    gradMw = np.zeros_like(Mw) #TODO
    gradM2 = np.outer(delta,childrenas)
    gradb2 = delta

    childrenads = np.concatenate([rnn.ad for rnn in self.rnns])
    # compute delta to backpropagate
    deltaB = np.split(np.multiply(np.transpose(M2).dot(delta),childrenads),2)
    # backpropagate to retrieve other gradients
    grad0M1, grad0b1, grad0V,grad0Mw= self.rnns[0].backprop(deltaB[0], M1,b1,V,Mw)
    grad1M1, grad1b1, grad1V,grad0Mw= self.rnns[1].backprop(deltaB[1], M1,b1,V,Mw)
    gradM1 = grad0M1 + grad1M1
    gradb1 = grad0b1 + grad1b1
    gradV  = grad0V  + grad1V
    return gradM1, gradb1, gradV, gradMw, gradM2, gradb2
  def __str__(self):
    return 't1: '+str(self.rnns[0])+', t2: '+str(self.rnns[1])



# activation functions:
def identity(vector):
  act = vector
  der = np.ones(len(act))
  return act, der
def tanh(vector):
  act = np.tanh(vector)
  der = 1- np.multiply(act,act)
  return act, der
def ReLU(vector):
  act = np.array([max(x,0)+0.01*min(x,0) for x in vector])
#  act = np.array([max(x,0) for x in vector])
  der = np.array([1*(x>=0) for x in vector])
  return act, der


'''
Helper functions to wrap and unwrap theta.
- wrap   input:  a tuple (M1,b1,V,Mw,M2,b2,M3,b3)
         output: a numpy array theta with the parameters
- unwrap input:  a numpy array theta with the parameters
         output: a tuple (M1,b1,V,M2,Mw,b2,M3,b3)
  NB: Unwrap assumes dint, dwords, nwords, dcomparison are instantiated as global variables.
'''
def wrap(parameters):

  theta = np.concatenate([np.reshape(par,-1) for par in parameters])
  print 'wrap', [str(np.shape(par)) for par in parameters],np.shape(theta)
  return theta
def unwrap(theta):
  print 'unwrap, theta shape=', np.shape(theta)
  left = 0
  right = left + dint*2*dint
  M1 = np.reshape(theta[left:right],(dint, 2*dint))
  left = right
  right = left + dint
  b1 = theta[left:right]
  left = right
  right = left+(nwords)*dint
  V = np.reshape(theta[left:right],(nwords, dint))
  left = right
  right = left+dint*dwords
  Mw = np.reshape(theta[left:right],(dint, dwords))
  left = right
  right = left + dcomparison*2*dint
  M2 = np.reshape(theta[left:right],(dcomparison, 2*dint))
  left = right
  right = left + dcomparison
  b2 = theta[left:right]
  left = right
  right = left + numrel*dcomparison
  M3 = np.reshape(theta[left:right],(numrel,dcomparison))
  left = right
  right = left + numrel
  b3 = theta[left:right]
  return M1,b1,V,Mw,M2,b2,M3,b3