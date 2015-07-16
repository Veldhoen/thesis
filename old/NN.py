from __future__ import division
import numpy as np
import sys

import activation


'''
 The NN class relies on a parameters object theta that is a numpy structured array.
   In this structure, the parameters can be retrieved on string-basis.
   The gradients returned after backpropagating the error, have a similar structure.

 A neural net consists of Node objects.
   In initialization, each node is assigned a category.
     This is used to identify the applicable parameters.
     A list of children, which should be node objects, can be assigned.
     Furthermore, the activation function ('identity','tanh','sigmoid',
     'ReLU' or 'softmax') must be provided.
   The function forward will recursively call the forward function of its
     children to perform a forward pass.
   The function backprop can be called to backpropagate an error message (delta)
     through the network and return gradients of the parameters.

 There are two specializations of Nodes: Leafs and Top.
   A Leaf stores additional information about the word and its index
     in the vocabulary.
     The backprop method updates the corresponding value
   A Top can be asked for a prediction and has a defined error function.
     The backprop method uses a target to compute an error message and
     backpropagates it through the network.
     This implementation assumes a softmax classifier as top layer,
     the error and backprop function should be modified for other tasks.

 The gradientCheck is a function to determine the corectness of the
   backpropagation. It computes a numerical gradient and compares it
   to the analytical gradient from the backprop. The difference should
   generally be smaller than 1e-7. However, this can vary: see
   http://cs231n.github.io/neural-networks-3/#gradcheck
'''

class Node:
  def __init__(self,children,cat, nonlinearity):
    self.cat = cat
    self.children = children
    self.nonlin = nonlinearity
  def backprop(self,delta,theta,gradients, recompute = False):
    childrenas = np.concatenate([child.a for child in self.children])
    childrenads = np.concatenate([child.ad for child in self.children])
    # compute delta to backprop and backprop it
    deltaB = np.split(np.multiply(np.transpose(theta[self.cat+'M']).dot(delta),childrenads),len(self.children))
    [self.children[i].backprop(deltaB[i], theta, gradients) for i in range(len(self.children))]
    # update gradients for this node
    gradients[self.cat+'M']+= np.outer(delta,childrenas)
    gradients[self.cat+'B']+=delta

  def activateNW(self, theta):
    self.forward(theta)

  def forward(self,theta):
    # recursively collect children's activation
    inputsignal = np.concatenate([child.forward(theta) for child in self.children])
    # compute activation to return
    M= theta[self.cat+'M']
    b= theta[self.cat+'B']
    self.z = M.dot(inputsignal)+b
    # store activation and its gradient for use in backprop
    self.a, self.ad = activation.activate(self.z,self.nonlin)
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
    self.a, self.ad = activation.activate(self.z,self.nonlin)
    return self.a
  def backprop(self,delta, theta, gradients):
    gradients[self.cat][self.index] += delta
  def __str__(self):
    return self.word

class Top(Node):
  def train(self, theta, target):
    self.forward(theta)
    return self.backprop(theta,target)


  def backprop(self,theta,target):
    # initialize gradients
    gradients = np.zeros_like(theta)
    # determine delta
    delta = np.array(self.a, copy = True)
    delta[target] -=1
    # call inherited backprop function
    Node.backprop(self,delta,theta, gradients)
    return gradients

  def error(self,theta, target, recompute = True):
    if recompute: self.forward(theta)
    return -np.log(self.a[target])

  def predict(self, theta, recompute = True):
    if recompute: self.forward(theta)
    return self.a.argmax(axis=0)
  def numericalGradient(theta, network, target):
    epsilon = 0.0001
    numgrad = np.zeros_like(theta)
    for name in theta.dtype.names:
    # create an iterator to iterate over the array, no matter its shape
      it = np.nditer(theta[name], flags=['multi_index'])
      while not it.finished:
        i = it.multi_index
        old = theta[name][i]
        # leaving all other parameters in place, add and subtract epsilon from this parameter
        # and determine the gradient using the network errors that result
        theta[name][i] = old + epsilon
        errorPlus = network.error(theta,target)
        theta[name][i] = old - epsilon
        errorMin = network.error(theta,target)
        d =(errorPlus-errorMin)/(2*epsilon)
        numgrad[name][i] = d
        theta[name][i] = old  # restore theta
        it.iternext()
    return numgrad

