from __future__ import division
from NN import *
import random
import sys
import numpy as np


def this2RNN(nltkTree):
  if nltkTree.height()>2:
    lhs = nltkTree.label()
    rhs = '('+ ', '.join([child.label() for child in nltkTree])+')'
    children = [this2RNN(t) for t in nltkTree]
    rnn = Node(children, [], ('composition',lhs,rhs,'I'), 'tanh')
  else:
    rnn = Leaf([],('word',), key=nltkTree[0],nonlinearity='identity')
  return rnn

def nodeLength(node):
  if isinstance(node,Leaf): return 1
  else: return sum([nodeLength(n) for n in node.inputs])

class RNN():
  def __init__(self, nltkTree):
    self.root =this2RNN(nltkTree)
    self.length = len(nltkTree.leaves())
#   def forward(self,theta, activateIn,activateOut):
#     self.root.forward(theta, activateIn,activateOut)
  def activate(self,theta):
    self.root.forward(theta, True, False)
    return self.root.a

  def length(self):
    try: return self.length
    except:
      self.length= nodeLength(self.root)
      return self.length





  def maxArity(self,node=None):
    if node is None: node = self
    ars = [self.maxArity(c) for c in node.inputs]
    ars.append(len(node.inputs))
    return max(ars)
  def __str__(self):
    return str(self.root)