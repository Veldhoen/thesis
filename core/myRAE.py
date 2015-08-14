from __future__ import division
from NN import *
import random
import sys
import numpy as np


import nltk # for now

# Reconstruction is a class that behaves like a node, but is actually a container of nodes

class Reconstruction(Node):

  def __init__(self,inputs, outputs, cat,nonlinearity, nodes, original):
    Node.__init__(self, inputs, outputs, cat,nonlinearity)
    self.nodes = nodes
    self.original = original

  def forward(self,inA,inAd, theta):
    self.inputsignal = inA
    self.dinputsignal = inAd
    Node.forward(self, theta, activateIn=False, activateOut = False, inputSet=True)    #set activation and its derivative for THIS

    lens = [int(np.shape(theta[c.cat+('M',)])[1]) for c in self.nodes] #but they are not activated yet!
    splitter = [sum(lens[:i]) for i in range(len(lens))][1:]
    for node, a, ad in zip(self.nodes, np.split(self.a,splitter), np.split(self.ad,splitter)):
      node.forward(a,ad,theta)


  def backprop(self, theta, gradient):
#    print 'backprop reconstruction'
    if len(self.nodes)>0:
      delta = np.concatenate([node.backprop(theta,gradient) for node in self.nodes])
    else:
      # backprop into original to intensify gradient for word matrix
      deltaR =  np.multiply(-(self.original.a-self.a),self.original.ad)
      self.original.backprop(theta,-1*deltaR,gradient,addOut=False)
      # determine error signal to backprop into the tree
      delta = np.multiply(-(self.original.a-self.a),self.ad)

    return Node.backprop(self,theta, delta, gradient, addOut = False, moveOn=False)
  def reconstructionError(self):
    if len(self.nodes)>0:
      return sum([node.reconstructionError() for node in self.nodes])

    else:
      length = np.linalg.norm(self.original.a-self.a)
      return .5*length*length

  def __str__(self):
    if self.original: return self.original.key+'-REC'
    return 'reconstruction('+','.join([str(node) for node in self.nodes])+')'

    #return 'reconstruction('+str(len(self.nodes))+' nodes, original = '+str(self.original)
def this2Rec(this, lhs, rhs):
  if len(this.inputs)>0:
    nodes = [this2Rec(node,lhs,rhs) for node in this.inputs]
    original = None
    cat = ('reconstruction',lhs,rhs,)
  else:
    nodes = []
    original = this
    cat = ('reconstructionLeaf',)
  rec = Reconstruction([this], [], cat, 'tanh',nodes, original)
  return rec

def this2RAE(nltkTree):
  if nltkTree.height()>2:
    lhs = nltkTree.label()
    rhs = '('+ ', '.join([child.label() for child in nltkTree])+')'

    children = [this2RAE(t) for t in nltkTree]

    rae = Node(children, [], ('composition',lhs,rhs,'I'), 'tanh')
#    [child.outputs.append(rae) for c in children]  # maybe leave this out, as outputs are mainly used for the reconstruction part?
    reconstruction = this2Rec(rae,lhs,rhs)
    rae.outputs = [reconstruction]

  else:
    rae = Leaf([],('word',), key=nltkTree[0],nonlinearity='identity')
  return rae
def nodeError(node):
  error = sum([n.reconstructionError() for n in node.outputs])
  error += sum([nodeError(n) for n in node.inputs])
#  print 'nodeError',node,error
  return error

class RAE():
  def __init__(self, nltkTree):
    self.root = this2RAE(nltkTree)
    
  def activate(self,theta):
#    print 'activate RAE'
    self.root.forward(theta, True, True)

  def train(self,theta, gradient, activate=True, target=None): #rain(self,theta,delta = None, gradient= None):
    if activate: self.activate(theta)
    if target is None: delta = np.zeros_like(self.root.a)
    else: delta = np.zeros_like(self.root.a)#True # make a delta message!
    self.root.backprop(theta, delta, gradient, addOut = True)


#   def error(self,theta,target = None):
#         self.forward(theta, True)
#         rootError = self.reconstructionError(theta)
#         return rootError + sum([child.error(theta) for child in self.children])
# 
#     # compute reconstruction error for this node: predict leafs and see how different they are from the actual leafs
#     def reconstructionError(self,theta, recalculate = True):
#         if len(self.children) == 0:
#            return 0
#         self.forward(theta,recalculate)
#         original = self.originalLeafs()
#         reconstruction = self.reconstruction.predictLeafs()
#         length = np.linalg.norm(original-reconstruction)
#         return .5*length*length

  def error(self,theta,target=None, activate=True):
    if activate: self.activate(theta)
    error = nodeError(self.root)
    if target is not None:
      length = np.linalg.norm(self.root.a-reconstruction)
      error += .5*length*length
    return error
# voc = ['UNKNOWN','most','large', 'hippos','bark','chase','dogs']
# gram = {'S':{'(NP, VP)':2},'NP':{'(Q, N)':2}}
# d = 3
# dims = {'inside':d,'outside':d,'word':d,'nwords':len(voc)}
# 
# theta = myTheta.Theta('RAE', dims,gram,None,voc)
# #  s = '(S (NP (Q most) (N hippos)) (VP (V chase) (NP (A big) (N dogs))))'
# s = '(S (NP (Q most) (N (A big) (N hippos))) (VP (V chase) (NP (A big) (N dogs))))'
# #  s = '(Top (S (NP (Q most) (N hippos)) (VP (V bark))))'
# #  s = '(NP (Q most) (N hippos))'
# #  s = '(Top (Q most))'
# #  s = '(Q most)'
# 
# thistree = nltk.tree.Tree.fromstring(s)
# network = RAE(thistree)
# network.activate(theta)
# gradient = network.train(theta)

#