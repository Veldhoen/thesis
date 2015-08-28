from __future__ import division
from NN import *
import random
import sys
import numpy as np

# Reconstruction is a class that behaves like a node, but is actually a container of nodes

class Reconstruction(Node):

  def __init__(self,inputs, outputs, cat,nonlinearity, nodes, original):
    Node.__init__(self, inputs, outputs, cat,nonlinearity)
    self.nodes = nodes
    self.original = original

  def aLen(self,theta):
    if len(self.nodes)>0:
      return int(np.shape(theta[self.cat+('M',)])[1])
    else: return self.original.aLen(theta)

  def forward(self,theta, activateIn = False, activateOut = False, signal=None):
#    print 'RAE.forward', self.cat, self.original#,len(signal[0])

    if signal is None: raise StandardError('Reconstruction.forward got no inputs'+str(self))

    if len(self.nodes)>0:
      # compute activation for this node, but no activateOut
      Node.forward(self, theta, activateIn=False, activateOut = False, signal=signal)    #set activation and its derivative for THIS

      #distribute the activation over the nodes
      lens = [c.aLen(theta) for c in self.nodes]
      splitter = [sum(lens[:i]) for i in range(len(lens))][1:]
#      print 'from:', self.cat, len(self.a), splitter
      for node, a, ad in zip(self.nodes, np.split(self.a,splitter), np.split(self.ad,splitter)):
#        print 'to:', node.cat,node, len(a)
        node.forward(theta, False, False, signal=(a,ad))
    else:
      # reconstruction leaf: do no computation at all
      self.inputsignal, self.dinputsignal = signal
      self.a, self.ad = signal

  def backprop(self, theta, delta, gradient,addOut=True,moveOn=False):
    if len(self.nodes)>0:
      if addOut:
        delta = np.concatenate([node.backprop(theta,None,gradient,addOut=True) for node in self.nodes])
        deltaB = Node.backprop(self,theta, delta, gradient, addOut = False, moveOn=False)
      else: raise RuntimeError('RAE.backprop, addOut is False')
    else:
      # backprop into original to intensify gradient for word matrix
      deltaR =  np.multiply(-(self.original.a-self.a),self.original.ad)
      self.original.backprop(theta,-1*deltaR,gradient,addOut=False,moveOn=False)
      # determine error signal to backprop into the tree
      deltaB = np.multiply(-(self.original.a-self.a),self.ad)
    return deltaB
  def reconstructionError(self):
    if len(self.nodes)>0:
      return sum([node.reconstructionError() for node in self.nodes])

    else:
      length = np.linalg.norm(self.original.a-self.a)
#      print length
      return .5*length*length

  def __str__(self):
    if self.original: return self.original.key+'-REC'
    return 'reconstruction['+','.join([str(node) for node in self.nodes])+']'

def this2Rec(this):
  if len(this.inputs)>0:
    nodes = [this2Rec(node) for node in this.inputs]
    original = None
    cat = ('reconstruction',this.cat[1],this.cat[2])
  else:
    nodes = []
    original = this
    cat = ('reconstruction','leaf')
  rec = Reconstruction([], [], cat, 'tanh',nodes, original)
  for node in nodes: node.inputs=[rec]
  return rec

def this2RAE(nltkTree):
  if nltkTree.height()>2:
    lhs = nltkTree.label()
    rhs = '('+ ', '.join([child.label() for child in nltkTree])+')'
    children = [this2RAE(t) for t in nltkTree]
    rae = Node(children, [], ('composition',lhs,rhs,'I'), 'tanh')
#    [child.outputs.append(rae) for c in children]  # maybe leave this out, as outputs are mainly used for the reconstruction part?
    reconstruction = this2Rec(rae)
    rae.outputs = [reconstruction]
    reconstruction.inputs=[rae]
  else:
    rae = Leaf([],('word',), key=nltkTree[0],nonlinearity='identity')
  return rae

def nodeError(node):

  errors = [n.reconstructionError() for n in node.outputs]
  errors += [nodeError(n) for n in node.inputs]
#  print 'nodeError',node,error
#  print 'nodeError of node:', node.cat, len(errors)

  try: return sum(errors)/len(errors)
  except: return 0

def nodeLength(node):
  if isinstance(node,Leaf): return 1
  else: return sum([nodeLength(n) for n in node.inputs])


class RAE():
  def __init__(self, nltkTree):
    self.root = this2RAE(nltkTree)
    self.length = len(nltkTree.leaves())

  def activate(self,theta):
    self.root.forward(theta, True, True)

  def length(self):
    return nodeLength(self.root)

  def train(self,theta, gradient, activate=True, target=None): #rain(self,theta,delta = None, gradient= None):
    if activate: self.activate(theta)
    if target is None: delta = np.zeros_like(self.root.a)
    else: delta = np.zeros_like(self.root.a)#True # make a delta message!
    self.root.backprop(theta, delta, gradient, addOut = True)
    return self.error(theta,None,False)

  def error(self,theta,target=None, activate=True):
    if activate: self.activate(theta)
    error = nodeError(self.root)
    if target is not None:
      length = np.linalg.norm(self.root.a-reconstruction)
      error += .5*length*length
    return error

  def evaluate(self,theta,sample):
    return self.error(theta)

  def maxArity(self,node=None):
    if node is None: node = self.root
    ars = [self.maxArity(c) for c in node.inputs]
    ars.append(len(node.inputs))
    return max(ars)
  def __str__(self):
    return str(self.root)