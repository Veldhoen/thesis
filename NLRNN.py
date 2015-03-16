from __future__ import division

from nltk import tree
import numpy as np

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
- error (theta, target):   return the error of the network with given parameters
'''

class Network:
  def __init__(self, comparisonlayer):
    self.comparison = comparisonlayer #

  def sethyperparams(self, nw, dw, dc, nr):
    global nwords, dwords, dcomparison,numrel
    nwords = nw
    dwords  = dw
    dcomparison = dc
    numrel = nr

  @classmethod
  def fromTrees(self,trees,words):
    return Network(Comparisonlayer([RNN(tree, words) for tree in trees]))

  def predict(theta):
    self.forward(theta)
    return a.argmax(axis=0)

  def forward(self, theta):
    M1,b1,V,M2,b2, M3, b3 = unwrap(theta)
    self.z = M3.dot(self.comparison.forward(M1,b1,V,M2,b2)) + b3
    self.a = np.exp(self.z) /    sum(np.exp(self.z))
    return self.a
  def backprop(self,theta,trueRelation):
    M1,b1,V,M2,b2, M3, b3 = unwrap(theta)
    # compute this node's delta
    delta = self.a
    delta[trueRelation] = -1+delta[trueRelation]   # Phong said 1-delta[trueRel], but that did not work
    # compute gradients for M3 and b3
    gradM3 = np.outer(delta,self.comparison.a)
    gradb3 = delta

    # compute delta to backpropagate
    deltaB = np.multiply(np.transpose(M3).dot(delta),self.comparison.ad)
    # backpropagate to retrieve other gradients
    gradM1, gradb1, gradV, gradM2, gradb2 = self.comparison.backprop(deltaB, M1,b1,V,M2,b2)
    return wrap((gradM1, gradb1, gradV, gradM2, gradb2, gradM3, gradb3))
  def error(self, theta, trueRelation):
    self.forward(theta)
    return -np.log(self.a[trueRelation])


class RNN:
  def __init__(self,tree,words):
    self.children = [RNN(child,words) for child in tree]
    try:    self.index = words.index(tree.label())  # Word index in V
    except: self.index = 0                          # Not a leaf or unknown word
  def forward(self, M,b,V):
    if len(self.children) > 0:
      self.z = M.dot(np.concatenate([child.forward(M,b,V) for child in self.children]))+b
      self.a, self.ad = tanh(self.z)
    else:
      self.z = V[self.index]
      self.a, self.ad = identity(self.z)
    return self.a
  def backprop(self, delta, M, b, V):
    if len(self.children) > 0:
      childrenas = np.concatenate([rnn.a for rnn in self.children])
      gradM = np.outer(delta,childrenas)
      gradb = delta

      childrenads = np.concatenate([rnn.ad for rnn in self.children])
      # compute delta to backpropagate
      deltaB = np.split(np.multiply(np.transpose(M).dot(delta),childrenads),2)
      # backpropagate to retrieve other gradients
      grad0M, grad0b, grad0V= self.children[0].backprop(deltaB[0], M,b,V)
      grad1M, grad1b, grad1V= self.children[1].backprop(deltaB[1], M,b,V)
      gradM += grad0M + grad1M
      gradb += grad0b + grad1b
      gradV  = grad0V  + grad1V
    else:
      gradM = np.zeros_like(M)
      gradb = np.zeros_like(b)
      gradV = np.zeros_like(V)
      gradV[self.index] = delta
    return gradM, gradb, gradV

class Comparisonlayer:
  def __init__(self,rnns):
    self.rnns = rnns
  def forward(self, M1,b1,V,M2,b2):
    self.z = M2.dot(np.concatenate([rnn.forward(M1,b1,V) for rnn in self.rnns]))+b2
    self.a, self.ad = ReLU(self.z)
    return self.a

  def backprop(self,delta, M1,b1,V,M2,b2):
    childrenas = np.concatenate([rnn.a for rnn in self.rnns])
    gradM2 = np.outer(delta,childrenas)
    gradb2 = delta

    childrenads = np.concatenate([rnn.ad for rnn in self.rnns])
    # compute delta to backpropagate
    deltaB = np.split(np.multiply(np.transpose(M2).dot(delta),childrenads),2)
    # backpropagate to retrieve other gradients
    grad0M1, grad0b1, grad0V= self.rnns[0].backprop(deltaB[0], M1,b1,V)
    grad1M1, grad1b1, grad1V= self.rnns[1].backprop(deltaB[1], M1,b1,V)
    gradM1 = grad0M1 + grad1M1
    gradb1 = grad0b1 + grad1b1
    gradV  = grad0V  + grad1V
    return gradM1, gradb1, gradV, gradM2, gradb2

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
  act = np.array([max(x,0) for x in vector])
  der = np.array([1*(x>=0) for x in vector])
  return act, der


'''
Helper functions to wrap and unwrap theta.
- wrap   input:  a tuple (M1,b1,V,M2,b2,M3,b3)
         output: a numpy array theta with the parameters
- unwrap input:  a numpy array theta with the parameters
         output: a tuple (M1,b1,V,M2,b2,M3,b3)
  NB: Unwrap assumes dwords, nwords, dcomparison are instantiated as global variables.
'''
def wrap(parameters):
  theta = np.concatenate([np.reshape(par,-1) for par in parameters])
  return theta
def unwrap(theta):
  left = 0
  right = left + dwords*2*dwords
  M1 = np.reshape(theta[left:right],(dwords, 2*dwords))
  left = right
  right = left + dwords
  b1 = theta[left:right]
  left = right
  right = left+(nwords)*dwords
  V = np.reshape(theta[left:right],(nwords, dwords))

  left = right
  right = left + dcomparison*2*dwords
  M2 = np.reshape(theta[left:right],(dcomparison, 2*dwords))
  left = right
  right = left + dcomparison
  b2 = theta[left:right]
  left = right
  right = left + numrel*dcomparison
  M3 = np.reshape(theta[left:right],(numrel,dcomparison))
  left = right
  right = left + numrel
  b3 = theta[left:right]
  return M1,b1,V,M2,b2,M3,b3