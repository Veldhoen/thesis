from __future__ import division
import activation
import numpy as np
import sys

import myTheta

class Node():
  def __init__(self,inputs, outputs, cat,nonlinearity):
#    print 'Node.init', cat, inputs
    self.inputs = inputs
    self.outputs = outputs
    self.cat = cat
    self.nonlin = nonlinearity

  def forward(self,theta, activateIn = True, activateOut = False):
#    print 'forward',self.cat[0], self.cat[-1] , self
    if activateIn:
  #    print 'do forward inputs'
      [i.forward(theta, activateIn,activateOut) for i in self.inputs]

    inputsignal = []
    for c in self.inputs:
      try: inputsignal+=[c.a]
      except: 
        print 'no activation in', c
        sys.exit()

    inputsignal = np.concatenate(inputsignal)
    #inputsignal = np.concatenate([c.a for c in self.inputs])


    M= theta[self.cat+('M',)]
    b= theta[self.cat+('B',)]
    if M is None or b is None:
      print 'Fail to forward node, no matrix and bias vector:', self.cat
      sys.exit()

    try: self.z = M.dot(inputsignal)+b
    except: print self.cat
    self.z = M.dot(inputsignal)+b

    self.a, self.ad = activation.activate(self.z, self.nonlin)
    if activateOut:
    #  print 'do forward outputs'
      [i.forward(theta, activateIn,activateOut) for i in self.outputs] #self.outputs.forward(theta, activateIn,activateOut)

  def backprop(self,theta, delta, gradient=None, addOut = False):
    if addOut: True #add a delta message from its outputs (e.g., reconstructions)


    if gradient is None: gradient = myTheta.gradient(theta) #theta.zeros_like(sparseWords = True)
    inputsignal = np.concatenate([c.a for c in inputs])
    dinputsignal = np.concatenate([c.ad for c in inputs])

    deltaB =np.multiply(np.transpose(theta[self.cat+'IM']).dot(delta),dinputsignal)
    lens = [len(c.a) for c in inputs]
    splitter = [sum(lens[:i]) for i in range(len(lens))]

    deltaBits = np.split(deltaB,splitter[:-1])
    [inputNode.backprop(theta, delt, gradient) for inputnode,delt in zip(self.inputs,deltaBits)]



    inputSignal= np.concatenate([child.innerA for child in self.children])
    childrenads = np.concatenate([child.innerAd for child in self.children])
    # compute delta to backprop and backprop it
    deltaB = np.split(np.multiply(np.transpose(theta[self.cat+'IM']).dot(delta),childrenads),len(self.children))
    [self.children[i].backpropInner(deltaB[i], theta, gradients) for i in xrange(len(self.children))]
    # update gradients for this node
    gradients[self.cat+'IM']+= np.outer(delta,childrenas)
    gradients[self.cat+'IB']+=delta

  def __str__(self):
    if self.cat[-1]=='I': return '('+self.cat[1]+' '+ ' '.join([str(child) for child in self.inputs])+')'
    if self.cat[-1]=='O': return '['+self.cat[1]+' '+ ' '.join([str(child) for child in self.inputs])+']' #'&'.join([str(c) for c in self.inputs])
    else: return '<'+self.cat[0]+' '+ ' '.join([str(child) for child in self.inputs])+'>'
    #return '['+self.cat[0]+' '+ ' '.join([str(child) for child in self.inputs])+']'


class Leaf(Node):
  def __init__(self,outputs,cat, key='',nonlinearity='identity'):
    Node.__init__(self,[], outputs, cat,nonlinearity)
    self.key = key

  def forward(self,theta, activateIn = True, activateOut = False):
#    print 'forward leaf',self.cat[0], self
    try: self.z = theta[self.cat][theta.lookup[self.cat].index(self.key)]
    except:
      print 'Fail to foward Leaf:', self.cat, self.key
      sys.exit()

    self.a, self.ad = activation.activate(self.z,self.nonlin)
    if activateOut:
#      print 'do forward outputs'
      [i.forward(theta, activateIn,activateOut) for i in self.outputs] #self.outputs.forward(theta, activateIn,activateOut)

  def __str__(self):
    return self.key#+'('+self.cat+')'